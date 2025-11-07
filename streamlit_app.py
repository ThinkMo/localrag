import streamlit as st
import json
import uuid
import requests
from typing import Optional, Iterator
import time


# app config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"  # default api address
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

class DocumentManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def upload_document(self, file, document_type: Optional[str] = None):
        """upload document"""
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
                st.error(f"upload document failed, status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"upload document failed: {str(e)}")
            return None
    
    def list_documents(self, skip: int = 0, page_size: int = 50, document_types: Optional[str] = None):
        """list documents"""
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
                st.error(f"list documents failed, status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"list documents failed: {str(e)}")
            return None
    
    def delete_document(self, document_id: int):
        """delete document"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/documents/{document_id}",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"delete document failed: {str(e)}")
            return False


def parse_sse_line(line):
    """
    Parse a single line from an SSE stream.
    event: message
    data: {"content": "Hello"}
    
    event: error
    data: {"error": "Something went wrong"}
    """
    line = line.strip()
    if not line:
        return None
    
    # "field: value"
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
        """send message stream"""
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
            yield "error: timeout"
        except Exception as e:
            yield f"error: {str(e)}"


def render_document_management():
    st.header("📁 document management")
    
    doc_manager = DocumentManager(st.session_state.api_base)
    
    # 创建标签页
    tab1, tab2 = st.tabs(["📤 upload document", "📋 document list"])
    
    with tab1:
        st.subheader("upload document")
        
        uploaded_file = st.file_uploader(
            "select document file",
            type=['pdf', 'md'],
            help="support pdf/md document format, max file size usually 100MB"
        )
        
        col, _  = st.columns(2)
        with col:
            doc_type = st.selectbox(
                "document type",
                ["pdf", "markdown"],
                key="doc_type"
            )
        
        if st.button("📤 upload document", type="primary", use_container_width=True) and uploaded_file:
            with st.spinner("uploading..."):
                result = doc_manager.upload_document(
                    uploaded_file, 
                    doc_type if doc_type else None
                )
                
            if result:
                st.success(f"✅ document '{uploaded_file.name}' uploaded successfully!")
                st.balloons()
                
                # show upload details
                if isinstance(result, dict):
                    with st.expander("upload details", expanded=False):
                        st.json(result)
            else:
                st.error("❌ upload document failed, please check api connection or file format")
    
    with tab2:
        st.subheader("document list")
        
        # 搜索和筛选选项
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 search document", placeholder="keyword...")
        with col2:
            doc_type_filter = st.selectbox(
                "filter document type",
                ["all types", "pdf", "markdown"]
            )
        with col3:
            if st.button("🔄 refresh list", use_container_width=True):
                st.rerun()
        
        # fetch document list
        with st.spinner("loading document list..."):
            documents = doc_manager.list_documents(
                document_types=doc_type_filter if doc_type_filter != "all types" else None
            )
        
        if documents and "items" in documents and len(documents["items"]) > 0:
            st.info(f"📊 found {len(documents['items'])} documents")
            
            for i, doc in enumerate(documents["items"]):
                # filter document name
                doc_name = doc.get('title', 'Unknown')
                if search_term and search_term.lower() not in doc_name.lower():
                    continue
                    
                with st.container():
                    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc_name}**")
                        doc_type = doc.get('document_type', 'unknown type')
                        st.caption(f"📝 type: {doc_type}")
                        
                        # show document metadata
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            if 'created_at' in doc:
                                st.caption(f"🕒 uploaded: {doc['created_at']}")
                        with meta_col2:
                            if 'size' in doc:
                                st.caption(f"📦 size: {doc['size']}")
                    
                    with col2:
                        doc_id = doc.get('id', 'N/A')
                        st.code(f"ID: {doc_id}")
                    
                    with col3:
                        if st.button("👁️ view", key=f"view_{doc_id}", use_container_width=True):
                            st.session_state[f"view_doc_{doc_id}"] = True
                    
                    with col4:
                        if st.button("🗑️ delete", key=f"delete_{doc_id}", use_container_width=True):
                            if doc_manager.delete_document(doc_id):
                                st.success("✅ document deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ failed to delete document")
                    
                    # show document details
                    if st.session_state.get(f"view_doc_{doc_id}", False):
                        with st.expander(f"document details: {doc_name}", expanded=True):
                            st.json(doc)
                            if st.button("close details", key=f"close_{doc_id}"):
                                st.session_state[f"view_doc_{doc_id}"] = False
                                st.rerun()
                    
                    if i < len(documents["items"]) - 1:
                        st.divider()
        else:
            st.info("📭 no documents available or failed to connect to document service")
            if st.button("upload document now", key="upload_from_empty"):
                st.switch_page("📤 upload document")


def render_chat_interface():
    st.header("💬 chat")
    
    chat_client = ChatClient(st.session_state.api_base)
    
    # sidebar settings
    with st.sidebar.expander("⚙️ settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.1,
                                  help="controls the randomness of the output")
            max_tokens = st.number_input("max tokens", 100, 4000, 1000,
                                       help="limits the maximum number of tokens in the output")
        with col2:
            top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.1,
                            help="controls the diversity of the output")
            presence_penalty = st.slider("presence penalty", -2.0, 2.0, 0.0, 0.1,
                                       help="controls the presence of new topics in the output")
        
        col3, col4 = st.columns(2)
        with col3:
            clear_history = st.button("clear history", use_container_width=True)
        with col4:
            export_chat = st.button("export chat", use_container_width=True)
        
        if clear_history:
            st.session_state.chat_history = []
            st.session_state.current_response = ""
            st.rerun()
        
        if export_chat:
            # export chat history
            chat_text = "chat history:\n\n"
            for msg in st.session_state.chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                chat_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                "download chat history",
                chat_text,
                file_name=f"chat_export_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
    
    # list chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # show timestamp if available
                if "timestamp" in message:
                    st.caption(f"🕒 {message['timestamp']}")
    
    # show current response if available
    if st.session_state.current_response:
        with st.chat_message("assistant"):
            st.write(st.session_state.current_response)
    
    # input area
    input_col1, input_col2 = st.columns([5, 1])
    with input_col1:
        prompt = st.chat_input("ask a question...")
    with input_col2:
        if st.button("🔄 new chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_response = ""
            st.session_state.context_id = str(uuid.uuid4())
            st.rerun()
    
    if prompt:
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            chat_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "history": st.session_state.chat_history[:-1]  # exclude current user message
            }
            
            try:
                for chunk in chat_client.send_message_stream(prompt, **chat_params):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"chat error: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg
        
        # add assistant message to chat history
        # st.session_state.current_response = full_response
        assistant_message = {
            "role": "assistant", 
            "content": full_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.chat_history.append(assistant_message)
        
        # scroll to bottom
        st.rerun()


def main():
    st.sidebar.title("🎯 settings")
    
    st.session_state.api_base = st.sidebar.text_input(
        "API base URL",
        value=st.session_state.api_base,
        help="e.g. http://localhost:8000 or https://your-api-domain.com"
    )
    st.session_state.context_id = str(uuid.uuid4())
    
    if st.sidebar.button("🔗 check connection", use_container_width=True):
        with st.spinner("testing connection..."):
            try:
                response = requests.get(
                    f"{st.session_state.api_base}/api/v1/documents/", 
                    timeout=5
                )
                if response.status_code == 200:
                    st.sidebar.success("✅ OK")
                else:
                    st.sidebar.error(f"❌ Connection failed: {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ Connection error: {str(e)}")
    
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio(
        "choose app mode",
        ["💬 chat", "📁 document management"],
        key="app_mode"
    )
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📖 instructions", expanded=True):
        st.markdown("""
        **document management features:**
        - upload multiple document formats
        - view and manage document list
        - delete unnecessary documents
        
        **chat features:**
        - streaming conversation
        - adjustable generation parameters
        - support for conversation history management
        
        **notes:**
        - ensure the backend API service is running
        - check network connection and API address
        - large file uploads may take some time
        """)
    
    # system status
    with st.sidebar.expander("🖥️ system status", expanded=False):
        st.metric("number of conversations", len(st.session_state.chat_history) // 2)
        st.metric("API address", st.session_state.api_base)
    
    # render app interface based on selected mode
    if app_mode == "📁 document management":
        render_document_management()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()