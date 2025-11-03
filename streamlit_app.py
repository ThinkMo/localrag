import streamlit as st
import json
import uuid
import requests
from typing import Optional, Iterator
import time


# 应用配置
st.set_page_config(
    page_title="文档管理与智能聊天",
    page_icon="🤖",
    layout="wide"
)

# 初始化session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"  # 默认API地址
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

class DocumentManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def upload_document(self, file, document_type: Optional[str] = None):
        """上传文档"""
        files = {"files": (file.name, file.getvalue(), file.type)}
        data = {}
        if document_type:
            data["document_type"] = document_type
            
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/documents/fileupload",
                files=files,
                data=data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"上传失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"上传失败: {str(e)}")
            return None
    
    def list_documents(self, skip: int = 0, page_size: int = 50, document_types: Optional[str] = None):
        """获取文档列表"""
        params = {
            "skip": skip,
            "page_size": page_size,
        }
        if document_types:
            params["document_types"] = document_types
            
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/documents/", 
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"获取文档列表失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"获取文档列表失败: {str(e)}")
            return None
    
    def delete_document(self, document_id: int):
        """删除文档"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/documents/{document_id}",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"删除失败: {str(e)}")
            return False


def parse_sse_line(line):
    """
    解析 Server-Sent Events 格式的行
    
    SSE 格式示例:
    event: message
    data: {"content": "Hello"}
    
    event: error
    data: {"error": "Something went wrong"}
    """
    line = line.strip()
    if not line:
        return None
    
    # SSE 格式通常是 "field: value"
    if ':' in line:
        field, value = line.split(':', 1)
        field = field.strip()
        value = value.strip()
        
        if field == 'data':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"raw_data": value}
        elif field == 'event':
            return {"event_type": value}
    
    return None


class ChatClient:
    def __init__(self, base_url: str):
        pass
    
    def send_message_stream(self, message: str, **kwargs) -> Iterator[str]:
        """发送聊天消息，返回流式响应"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache',
            }
            payload = {
                "id": str(uuid.uuid4()),
                "jsonrpc": "2.0",
                "method": "message/stream",
                "params": {
                    "message": {
                        "contextId": st.session_state.context_id,
                        "kind": "message",
                        "messageId": "string",
                        "parts": [{
                            "kind": "text",
                            "text": message
                        }],
                        "role": "agent",
                    },
                }
            }
            response = requests.post(
                f"{st.session_state.api_base}/a2a",
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    event_data = parse_sse_line(line)
                    if event_data and isinstance(event_data, dict):  
                        if "result" in event_data:
                            if "artifact" in event_data["result"]:
                                if "parts" in event_data["result"]["artifact"]:
                                    for part in event_data["result"]["artifact"]["parts"]:
                                        if part["kind"] == "text":
                                            yield part["text"]
  
        except requests.exceptions.Timeout:
            yield "错误: 请求超时，请稍后重试"
        except Exception as e:
            yield f"连接错误: {str(e)}"


def render_document_management():
    st.header("📁 文档管理")
    
    doc_manager = DocumentManager(st.session_state.api_base)
    
    # 创建标签页
    tab1, tab2 = st.tabs(["📤 上传文档", "📋 文档列表"])
    
    with tab1:
        st.subheader("上传新文档")
        
        uploaded_file = st.file_uploader(
            "选择文档文件",
            type=['pdf', 'md'],
            help="支持pdf/md文档格式，最大文件大小通常为100MB"
        )
        
        col, _  = st.columns(2)
        with col:
            doc_type = st.selectbox(
                "文档类型",
                ["pdf", "markdown"],
                key="doc_type"
            )
        
        if st.button("📤 上传文档", type="primary", use_container_width=True) and uploaded_file:
            with st.spinner("上传中..."):
                result = doc_manager.upload_document(
                    uploaded_file, 
                    doc_type if doc_type else None
                )
                
            if result:
                st.success(f"✅ 文档 '{uploaded_file.name}' 上传成功！")
                st.balloons()
                
                # 显示上传结果信息
                if isinstance(result, dict):
                    with st.expander("上传详情", expanded=False):
                        st.json(result)
            else:
                st.error("❌ 文档上传失败，请检查API连接或文件格式")
    
    with tab2:
        st.subheader("文档列表")
        
        # 搜索和筛选选项
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 搜索文档", placeholder="输入文档名称关键词...")
        with col2:
            doc_type_filter = st.selectbox(
                "筛选类型",
                ["所有类型", "pdf", "markdown"]
            )
        with col3:
            if st.button("🔄 刷新列表", use_container_width=True):
                st.rerun()
        
        # 获取文档列表
        with st.spinner("加载文档列表中..."):
            documents = doc_manager.list_documents(
                document_types=doc_type_filter if doc_type_filter != "所有类型" else None
            )
        
        if documents and "items" in documents and len(documents["items"]) > 0:
            st.info(f"📊 找到 {len(documents['items'])} 个文档")
            
            for i, doc in enumerate(documents["items"]):
                # 文档名称过滤
                doc_name = doc.get('title', 'Unknown')
                if search_term and search_term.lower() not in doc_name.lower():
                    continue
                    
                with st.container():
                    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc_name}**")
                        doc_type = doc.get('document_type', '未知类型')
                        st.caption(f"📝 类型: {doc_type}")
                        
                        # 显示文档元数据
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            if 'created_at' in doc:
                                st.caption(f"🕒 上传: {doc['created_at']}")
                        with meta_col2:
                            if 'size' in doc:
                                st.caption(f"📦 大小: {doc['size']}")
                    
                    with col2:
                        doc_id = doc.get('id', 'N/A')
                        st.code(f"ID: {doc_id}")
                    
                    with col3:
                        if st.button("👁️ 查看", key=f"view_{doc_id}", use_container_width=True):
                            st.session_state[f"view_doc_{doc_id}"] = True
                    
                    with col4:
                        if st.button("🗑️ 删除", key=f"delete_{doc_id}", use_container_width=True):
                            if doc_manager.delete_document(doc_id):
                                st.success("✅ 文档删除成功！")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 删除失败")
                    
                    # 查看文档详情
                    if st.session_state.get(f"view_doc_{doc_id}", False):
                        with st.expander(f"文档详情: {doc_name}", expanded=True):
                            st.json(doc)
                            if st.button("关闭详情", key=f"close_{doc_id}"):
                                st.session_state[f"view_doc_{doc_id}"] = False
                                st.rerun()
                    
                    if i < len(documents["items"]) - 1:
                        st.divider()
        else:
            st.info("📭 暂无文档或无法连接到文档服务")
            if st.button("立即上传文档", key="upload_from_empty"):
                st.switch_page("📤 上传文档")


def render_chat_interface():
    st.header("💬 智能聊天")
    
    chat_client = ChatClient(st.session_state.api_base)
    
    # 侧边栏聊天设置
    with st.sidebar.expander("⚙️ 聊天设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("创造性", 0.0, 2.0, 0.7, 0.1,
                                  help="值越高，回答越有创造性")
            max_tokens = st.number_input("最大长度", 100, 4000, 1000,
                                       help="限制生成文本的最大长度")
        with col2:
            top_p = st.slider("核心采样", 0.1, 1.0, 0.9, 0.1,
                            help="控制生成的多样性")
            presence_penalty = st.slider("话题新鲜度", -2.0, 2.0, 0.0, 0.1,
                                       help="避免重复已提及的内容")
        
        col3, col4 = st.columns(2)
        with col3:
            clear_history = st.button("清空历史", use_container_width=True)
        with col4:
            export_chat = st.button("导出对话", use_container_width=True)
        
        if clear_history:
            st.session_state.chat_history = []
            st.session_state.current_response = ""
            st.rerun()
        
        if export_chat:
            # 简单的对话导出功能
            chat_text = "对话记录:\n\n"
            for msg in st.session_state.chat_history:
                role = "用户" if msg["role"] == "user" else "助手"
                chat_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                "下载对话记录",
                chat_text,
                file_name=f"chat_export_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # 为每条消息添加时间戳（如果可用）
                if "timestamp" in message:
                    st.caption(f"时间: {message['timestamp']}")
    
    # 如果当前有正在生成的响应，显示它
    if st.session_state.current_response:
        with st.chat_message("assistant"):
            st.write(st.session_state.current_response)
    
    # 聊天输入区域
    input_col1, input_col2 = st.columns([5, 1])
    with input_col1:
        prompt = st.chat_input("输入您的问题或指令...")
    with input_col2:
        if st.button("🔄 新对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_response = ""
            st.session_state.context_id = str(uuid.uuid4())
            st.rerun()
    
    if prompt:
        # 添加用户消息到历史
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_history.append(user_message)
        
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)
        
        # 获取AI回复（流式）
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            chat_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "history": st.session_state.chat_history[:-1]  # 排除当前消息
            }
            
            try:
                for chunk in chat_client.send_message_stream(prompt, **chat_params):
                    if chunk:
                        # 累积响应
                        full_response += chunk
                        
                        # 更新显示（带打字机效果）
                        message_placeholder.markdown(full_response + "▌")
                
                # 流式完成，移除光标
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"聊天出错: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg
        
        # 更新当前响应和聊天历史
        # st.session_state.current_response = full_response
        assistant_message = {
            "role": "assistant", 
            "content": full_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_history.append(assistant_message)
        
        # 自动滚动到底部
        st.rerun()


def main():
    st.sidebar.title("🎯 配置设置")
    
    # API配置
    st.session_state.api_base = st.sidebar.text_input(
        "API基础地址",
        value=st.session_state.api_base,
        help="例如: http://localhost:8000 或 https://your-api-domain.com"
    )
    st.session_state.context_id = str(uuid.uuid4())
    
    # 连接测试
    if st.sidebar.button("🔗 测试连接", use_container_width=True):
        with st.spinner("测试连接中..."):
            try:
                # 测试文档接口
                response = requests.get(
                    f"{st.session_state.api_base}/api/v1/documents/", 
                    timeout=5
                )
                if response.status_code == 200:
                    st.sidebar.success("✅ 连接成功")
                else:
                    st.sidebar.error(f"❌ 连接失败，状态码: {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ 连接错误: {str(e)}")
    
    # 功能导航
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio(
        "选择功能",
        ["💬 智能聊天", "📁 文档管理"],
        key="app_mode"
    )
    
    st.sidebar.markdown("---")
    
    # 使用说明
    with st.sidebar.expander("📖 使用说明", expanded=True):
        st.markdown("""
        **文档管理功能:**
        - 上传多种格式的文档
        - 查看和管理文档列表
        - 删除不需要的文档
        
        **智能聊天功能:**
        - 流式对话
        - 可调整生成参数
        - 支持对话历史管理
        
        **注意事项:**
        - 确保后端API服务正在运行
        - 检查网络连接和API地址
        - 大文件上传可能需要较长时间
        """)
    
    # 系统状态
    with st.sidebar.expander("🖥️ 系统状态", expanded=False):
        st.metric("对话轮数", len(st.session_state.chat_history) // 2)
        st.metric("API地址", st.session_state.api_base)
    
    # 根据选择显示相应界面
    if app_mode == "📁 文档管理":
        render_document_management()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()