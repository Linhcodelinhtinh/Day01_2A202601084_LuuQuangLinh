# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua bốn lần thử, em nhận thấy khi temperature tăng lên, mức độ đa dạng của việc chọn từ ngữ tăng lên, độ mạch lạc và tính chính xác có phần giảm. Khi temperature đặt ở mức 0.0 và 0.7, câu trả lời chính xác và mạch lạc. Ở mức 1.2, câu trả lời vẫn chưa quá khác biệt. Khi tăng lên 1.8, phản hồi có dấu hiệu hơi bay bổng, phóng khoáng quá mức và thiếu nghiêm túc. Điều này là dễ hiểu do khi tăng temp, log prob của token được chia cho 1 mẫu số lớn hơn, khiến sự chênh lệch xác suất xuất hiện của các từ/token không còn quá lớn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> - Trợ lý soạn thảo hợp đồng pháp lý: Đặt temperature thấp, khoảng 0.0 đến 0.2 để tính chính xác tuyệt đối, chuẩn mực về ngôn từ và không bịa đặt điều khoản. Điều này giúp phản hồi có tính nhất quán và độ tin cậy cao nhất. Trợ lý viết slogan quảng cáo: Đặt temperature trung bình đến cao, khoảng 1 đến 1.5 để câu từ có sự sáng tạo và khác những khuôn mẫu thông thường.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> - Tổng số token mỗi ngày: 20.000 user * 2 lần/ngày * 500 token = 20 triệu token. Chi phí tham khảo với gpt-4o sẽ rơi vào khoảng 215 usd/ngày và chi phí với gpt-4o-mini khoảng 12 usd/ngày.
> - Model lớn xứng đáng khi cần giải quyết các bài toán lập luận logic phức tạp, sửa lỗi lập trình hệ thống khó hoặc phân tích số liệu tài chính/pháp lý chuyên sâu cần độ chuẩn xác cao. Lựa chọn model nhỏ với tốc độ phản hồi nhanh và chi phí rẻ với các subtask đơn giản như phân loại cảm xúc văn bản, tóm tắt văn bản ngắn hay lọc từ khóa.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> - Phản hồi của nhà thơ mang giọng văn bay bổng, sử dụng nhiều hình ảnh ví von (như hạt mầm, đất mẹ), tuân thủ yêu cầu không dùng thuật ngữ kỹ thuật, độ dài vừa phải. Phản hồi của kỹ sư senior lại rất chính xác, cấu trúc logic rõ ràng với các gạch đầu dòng, sử dụng thuật ngữ chuyên môn và có kèm theo ví dụ code để minh họa.
> - System prompt có thể điều khiển các khía cạnh: Phong cách, định dạng/cấu trúc phản hồi, độ dài/ngắn của câu trả lời, mức độ chuyên sâu kỹ thuật, và các ràng buộc/quy tắc ứng xử.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> - Đối với đoạn văn tiếng Việt ~137 từ ở trên, ước lượng thô là 137 / 0.75 = 182.6 token, trong khi thực tế đếm bằng tiktoken (gpt-4o) là 192 token, chênh lệch khoảng 5% (thực tế nhiều hơn ước lượng). Với các tokenizer cũ hơn như của gpt-4, số token thực tế là 340 token, chênh lệch tới ~86% so với ước lượng thô.
> - Nếu dùng ước lượng thô (1 từ = 1.33 token) để dự toán ngân sách cho ứng dụng tiếng Việt, ta sẽ dự toán thiếu. Lý do là vì tiếng Việt là ngôn ngữ đa âm tiết, có nhiều dấu thanh và các ký tự đặc biệt, khiến bộ mã hóa của LLM phải phân tách từ thành nhiều sub-word token hơn so với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming vì người dùng có thể đọc câu trả lời ngay khi model đang sinh nội dung, giúp giảm cảm giác chờ đợi và cải thiện UX rõ rệt. Trái lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) hoàn toàn không cần streaming vì đây là tác vụ xử lý hàng loạt chạy ngầm, không có người dùng tương tác trực tiếp theo thời gian thực nên chỉ cần nhận kết quả hoàn chỉnh cuối cùng. Trợ lý giọng nói đọc to phản hồi (b) là trường hợp trung gian, có thể tận dụng stream để xử lý văn bản thành giọng nói theo từng câu nhằm giảm độ trễ trước khi model hoàn thành toàn bộ câu trả lời, nhưng việc tích hợp sẽ phức tạp hơn chatbot văn bản.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> - Exponential backoff giúp tăng thời gian chờ giữa các lần thử lại sau mỗi lần thất bại theo cấp số nhân (ví dụ: 0.1s, 0.2s, 0.4s...). So với delay cố định, kỹ thuật này giúp giảm nhanh và đáng kể tần suất yêu cầu gửi đến máy chủ, giúp hệ thống có thời gian tự phục hồi khi bị quá tải.
> - Kỹ thuật "jitter" (thêm độ trễ ngẫu nhiên) giúp giải quyết vấn đề các client cùng bị lỗi sẽ tính toán thời gian thử lại trùng nhau và cùng gửi yêu cầu lên máy chủ tại cùng một thời điểm, tạo ra sức ép lớn lên máy chủ. Jitter giúp phân tán thời điểm gửi yêu cầu của các client một cách ngẫu nhiên, làm phẳng biểu đồ tải trọng trên máy chủ.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> - System prompt ví dụ: "Bạn là một trợ lý giảng dạy AI chuyên nghiệp, tận tâm và thân thiện. Hãy giải thích các khái niệm một cách ngắn gọn, dễ hiểu và luôn cung cấp ví dụ code minh họa bằng Python khi giải thích về kỹ thuật. Luôn phản hồi bằng tiếng Việt."
> - Hai vị trí thay đổi hành vi rõ rệt khi xóa:
>   1. "luôn cung cấp ví dụ code minh họa bằng Python khi giải thích về kỹ thuật": Nếu xóa đi, trợ lý sẽ có xu hướng giải thích lý thuyết thuần túy và không cung cấp code mẫu nữa, khiến học viên gặp khó khăn khi thực hành.
>   2. "Luôn phản hồi bằng tiếng Việt": Nếu xóa đi, khi học viên gửi câu hỏi hoặc mã lỗi bằng tiếng Anh, trợ lý có thể tự động trả lời bằng tiếng Anh thay vì tiếng Việt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> - Tình huống mất ngữ cảnh: Khi học viên nhờ trợ lý debug một đoạn code lớn ở lượt 1, sau đó hỏi tiếp 3 câu hỏi phụ ngắn ở lượt 2, 3, 4. Đến lượt 5, học viên hỏi: *"Hãy viết lại đoạn code ban đầu sau khi đã sửa các lỗi trên."* Lúc này, do history chỉ giữ tối đa 4 lượt cuối (lượt 1 đã bị cắt bỏ), trợ lý sẽ không còn giữ đoạn code gốc và không thể viết lại được.
> - Đề xuất khắc phục: Sử dụng cơ chế tóm tắt ngữ cảnh. Khi history vượt quá giới hạn (ví dụ > 4 lượt), ta dùng một model nhỏ để tóm tắt các thông tin quan trọng của các lượt hội thoại cũ (như đoạn code gốc, các lỗi đã xác định) thành một đoạn tóm tắt ngắn, sau đó đính kèm đoạn tóm tắt này vào system prompt hoặc lượt chat đầu tiên của history mới để duy trì ngữ cảnh dài hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
