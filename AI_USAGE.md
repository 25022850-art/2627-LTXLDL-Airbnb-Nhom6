1. Lần 1:
    - Công cụ: Gemini
    - Dùng vào việc: Hỗ trợ lấy thư mục gốc của dự án khi đang ở trong file download.py.
    - AI sai ở đâu: Không biết rõ cấp bậc của thư mục (do mình không cung cấp rõ ngữ cảnh).
    - Kiểm chứng: Lưu thử 1 file tại đường dẫn tạo ra từ code của AI và kiểm tra xem file đó có xuất hiện và chứa nội dung vừa ghi hay không.

2. Lần 2:
    - Công cụ: Gemini
    - Dùng vào việc: Fix lỗi ở phần thông báo lỗi của các lệnh assert trong file download.py (xuống dòng cần thêm '\' để báo cho Python biết câu lệnh còn tiếp tục ở dưới).
    - AI sai ở đâu: Cái này chỉ là thêm kí tự '\' để đúng syntax nên tôi cũng không biết AI có thể sai ở đâu.
    - Kiểm chứng: Sửa đổi vài tham số nhằm gây ra lỗi khi assert để xem có thông báo ra lỗi đã ghi trước không.

3. Lần 3:
    - Công cụ: Claude
    - Dùng vào việc: Hỗ trợ viết khung mẫu cho qa.py (gồm các hàm _flag_field, id_flags, _qa_numeric_field, _qa_category_field, clean_listings)
    - AI sai ở đâu: Có thể hiểu sai quy tắc QA của nhóm vì AI không biết ngữ cảnh dữ liệu.
    - Kiểm chứng: Kiểm tra đầu ra của report QA có đủ và đúng với các vấn dề bắt được khi chạy trong notebook hay không. Đọc lại để xem AI đã xử lý các cột thiếu như nào, có đúng với mong muốn không.

4. Lần 4: 
    - Công cụ: Gemini
    - Dùng vào việc: Hỗ trợ viết hàm kiểm tra cấu trúc dữ liệu (kiem_tra_cau_truc_all) trong qa.py.
    - AI sai ở đâu: Dùng is_numeric_dtype để kiểm tra cả cột id, trong khi id phải là kiểu chuỗi nên kiểm tra này không phù hợp.
    - Kiểm chứng: Đối chiếu với kiểu dữ liệu thật của cột id và sửa lại. Kiểm tra xem AI đã xét đầy đủ các cột cần thiết chưa và chạy thử trên 1 snapshot, và thử bỏ 1 cột có trong yêu cầu đi xem đoạn code có bắt được lỗi không.

5. Lần 5:
    - Công cụ: Claude
    - Dùng vào việc: Hỗ trợ tách phần kiểm tra giá trị price ngoại lại thành hàm riêng (price_bounds) trong qa.py.
    - AI sai ở đâu: Vô tình sửa dữ liệu gốc, quên không gọi lại hàm sau khi tách.
    - Kiểm chứng: Kiểm tra code xem price_bounds đã được chưa, chạy thử trên dữ liệu mẫu để đối chiếu dữ liệu gốc trước và sau khi gọi hàm, và kiểm tra báo cáo bắt giá ngoại lai vẫn giống như trước khi tách.

6. Lần 6:
    - Công cụ: Gemini
    - Dùng vào việc: Cung cấp các Regex Pattern cho các aspects (cả tích cực, tiêu cực, mixed/neutral/acceptable).
    - AI sai ở đâu: Thiếu ranh giới từ \b. 1 pattern xuất hiện ở nhiều nhãn khác nhau đối lập nhau.
    - Kiểm chứng: Kiểm tra và bổ sung \b cho các pattern, rà soát để loại bỏ các pattern bị trùng giữa các nhãn đối lập, sau đó chạy thử trên tập mẫu (nhãn tay) để xem bắt nhầm hay bỏ sót ở đâu.

7. Lần 7:
    - Công cụ: Claude
    - Dùng vào việc: Tạo ra mẫu nhỏ để kiểm thử xem code chạy đúng không so với các công thức tính KPI định trước (AI tạo ra cả đáp án).
    - AI sai ở đâu: Có thể hiểu sai bản chất công thức, không nắm rõ bản chất của dữ liệu của nhóm để tạo ra mẫu phù hợp, hoặc tạo ra đáp án sai.
    - Kiểm chứng: Tính tay lại các KPI để xác nhận đáp án AI đưa ra là chính xác, đọc lại dữ liệu AI tạo ra để xem có phù hợp với cách nhóm xử lý với dữ liệu của nhóm không (ví dụ: bỏ dòng thiếu giá).

8. Lần 8:
    - Công cụ: Claude
    - Dùng vào việc: Sử dụng các hàm trong QA, KPI (không sửa logic, chỉ gọi hàm) để code ra đoạn xử lý và tính toán trên mẫu nhỏ kiểm thử.
    - AI sai ở đâu: Truyền sai tham số so với dự định, đường dẫn import sai do không biết về thứ bậc folder.
    - Kiểm chứng: Kiểm tra lại tham số truyền vào, đường dẫn import theo cấu trúc thư mục và luồng gọi các hàm. Chạy trực tiếp đoạn code trên mẫu nhỏ để đảm bảo không báo lỗi và ra kết quả cần kiểm tra.