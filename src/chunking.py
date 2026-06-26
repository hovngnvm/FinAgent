import re

def preserve_markdown_tables(text: str) -> list:
    """
    Hàm nhận diện và cô lập các bảng biểu dạng Markdown.
    Đảm bảo bảng số liệu tài chính không bị cắt đôi bừa bãi.
    """
    # Regex nhận diện khối bảng Markdown tiêu chuẩn
    table_regex = re.compile(r'(?:\|[^\n]+\|\n\|(?:[\s]*:?-+:?[\s]*\|)+\n(?:\|[^\n]+\|\n*)+)')
    
    tables = table_regex.findall(text)
    # Thay thế tạm thời bảng bằng một token định danh để không bị can thiệp lúc split chuỗi
    placeholder_text = text
    for idx, table in enumerate(tables):
        placeholder_text = placeholder_text.replace(table, f"__TABLE_PLACEHOLDER_{idx}__")
        
    return placeholder_text, tables

def advanced_parent_child_chunker(text: str, source_link: str, parent_size: int = 1200, child_size: int = 250) -> list[dict]:
    """
    Cấu hình bộ phân tách dữ liệu Phân cấp (Parent-Child Chunking)
    Bảo vệ nguyên vẹn ngữ cảnh của câu và bảng biểu tài chính.
    """
    # Bước 1: Cô lập bảng biểu tài chính
    processed_text, preserved_tables = preserve_markdown_tables(text)
    
    # Bước 2: Tách đoạn thô theo dấu ngắt dòng câu
    paragraphs = [p.strip() for p in processed_text.split("\n\n") if p.strip()]
    
    parent_chunks = []
    current_parent = []
    current_length = 0
    
    # Gom các phân đoạn nhỏ thành các khối Parent Context lớn (~1200 ký tự/tokens)
    for para in paragraphs:
        current_parent.append(para)
        current_length += len(para)
        
        if current_length >= parent_size:
            parent_chunks.append("\n\n".join(current_parent))
            current_parent = []
            current_length = 0
            
    if current_parent:
        parent_chunks.append("\n\n".join(current_parent))
        
    final_prepared_payloads = []
    
    # Bước 3: Phân rã từng Parent lớn thành các Child Chunks nhỏ (~250 ký tự) để lấy index vector
    for p_idx, parent_content in enumerate(parent_chunks):
        # Khôi phục lại dữ liệu bảng thật vào trong Parent Chunk nếu chứa placeholder
        actual_parent_text = parent_content
        for t_idx, table_content in enumerate(preserved_tables):
            actual_parent_text = actual_parent_text.replace(f"__TABLE_PLACEHOLDER_{t_idx}__", table_content)
            
        # Chia nhỏ Child theo dấu chấm câu hoặc khoảng trắng
        sentences = re.split(r'(?<=[.!?])\s+', actual_parent_text)
        
        current_child = []
        current_child_len = 0
        
        for sentence in sentences:
            current_child.append(sentence)
            current_child_len += len(sentence)
            
            if current_child_len >= child_size:
                child_text = " ".join(current_child).strip()
                if child_text:
                    final_prepared_payloads.append({
                        "text": child_text,                 # Dùng để sinh nhãn Dense/Sparse Vector (Child)
                        "parent_text": actual_parent_text,   # Context thực tế sẽ đẩy vào prompt LLM (Parent)
                        "source": source_link,
                        "chunk_hierarchy": f"p{p_idx}-c{len(final_prepared_payloads)}"
                    })
                current_child = []
                current_child_len = 0
                
        # Gom nốt phần đuôi Child còn sót lại của khối Parent đó
        if current_child:
            child_text = " ".join(current_child).strip()
            if child_text:
                final_prepared_payloads.append({
                    "text": child_text,
                    "parent_text": actual_parent_text,
                    "source": source_link,
                    "chunk_hierarchy": f"p{p_idx}-tail"
                })
                
    return final_prepared_payloads