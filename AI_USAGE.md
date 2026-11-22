###. Rà soát xử lý dữ liệu và kết quả:
- Công cụ: ChatGPT/Codex.
- Dùng vào việc: Hỗ trợ kiểm tra dữ liệu thiếu, sai định dạng, ID trùng; phân biệt điểm đánh giá bị thiếu với trường hợp chưa có review.
- Hạn chế: Chuyển dữ liệu bằng to_numeric có thể khiến ô thiếu và chuỗi sai cùng thành NaN, nên cần kiểm tra giá trị gốc.
- Kiểm chứng: Thử các trường hợp riêng và đối chiếu bảng QA, KPI trước/sau chỉnh sửa bằng code trong phiên hỗ trợ.
