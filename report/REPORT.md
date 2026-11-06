# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về **cả bức ảnh**. Vì đây là tác vụ Image Classification (phân loại ảnh), mô hình chỉ dự đoán một nhãn đại diện duy nhất cho toàn bộ khung hình kèm xác suất, hoàn toàn không có thông tin tọa độ vị trí hay bounding box của từng phương tiện riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp **đi kèm taxonomy/checkpoint** (cụ thể là tập nhãn chuẩn ImageNet-1K gồm 1000 lớp được đóng gói sẵn trong file trọng số `yolo11n-cls.pt`), mô hình không tự nghĩ ra nhãn.
4. ì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì mỗi trường phục vụ một mục đích khác nhau:
- `class_id`: Định danh dạng số nguyên để máy tính và pipeline xử lý thuật toán, tính toán loss/metric.
- `class_name`: Tên lớp dạng văn bản giúp con người đọc hiểu và kiểm tra trực quan.
- `taxonomy_name`: Xác định hệ quy chiếu phân loại (ImageNet-1K), tránh nhầm lẫn hoặc xung đột ý nghĩa nhãn khi chuyển đổi sang các tập dữ liệu khác (như COCO hay Pascal VOC).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ **quy tắc ưu tiên (priority rules)**, chẳng hạn: chọn theo chủ thể chiếm diện tích lớn nhất (dominant subject), đối tượng nằm ở trọng tâm tiền cảnh, hoặc ưu tiên theo lớp mục tiêu nghiệp vụ cụ thể.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì `score` (ở đây là 0.510915) chỉ là xác suất dự đoán thống kê của mô hình (prediction), mô hình hoàn toàn có thể đoán sai hoặc thiếu chắc chắn (như ảnh thực tế có nhiều xe buýt và ô tô con nhưng model chỉ đoán "cab" với độ tin cậy ~51%). Ground truth phải là nhãn đúng tuyệt đối do con người xác thực dựa theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box bao quanh người đứng ở phía bên phải của ảnh.
- Góc trên-trái: tọa độ (385.33, 69.24), nằm gần phía trên và bên phải ảnh.
- Góc dưới-phải: tọa độ (498.92, 348.92), nằm ở phía dưới người.
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
Như vậy, box có chiều rộng khoảng 114 pixel và chiều cao khoảng 280 pixel, bao quanh gần như toàn bộ cơ thể người trong ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 
Chọn hai ngưỡng 0.35 và 0.60. 
- Ngưỡng 0.35: giữ lại 11 prediction có score từ 0.35 trở lên.
- Ngưỡng 0.60: chỉ giữ lại 6 prediction có score từ 0.60 trở lên.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ confidence threshold, hệ thống giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp.
Vì vậy, reviewer phải:
- Kiểm tra thêm các bounding box mới được giữ lại.
- Xác minh đối tượng được phát hiện có chính xác hay không.
- Phát hiện và loại bỏ những box sai, box trùng lặp hoặc nhận diện nhầm lớp.
- Kiểm tra lại vị trí và kích thước bounding box.
Điều này làm tăng khối lượng công việc và thời gian review, nhưng giúp hạn chế bỏ sót đối tượng thực tế.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc: Người gán nhãn phải vẽ bounding box sát với đường biên của đối tượng, bao phủ toàn bộ phần đối tượng nhìn thấy được, không lấy thừa vùng nền xung quanh. Nếu đối tượng bị che khuất hoặc bị cắt khỏi mép ảnh, chỉ vẽ box theo phần đối tượng thực sự nhìn thấy, không tự suy đoán phần bị khuất.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, chiếc lò nướng ở phía bên trái ảnh bị cắt một phần bởi mép trái của khung hình.
Cách xử lý: Mình chọn đưa trường hợp này thành quy tắc trong guideline.
Lý do: Đây là tình huống có thể gặp thường xuyên trong nhiều ảnh. Người gán nhãn chỉ cần vẽ bounding box bao quanh phần lò nướng còn nhìn thấy trong ảnh, không mở rộng box ra ngoài ranh giới hình ảnh.
Việc bổ sung quy tắc này vào guideline giúp người gán nhãn thực hiện thống nhất, hạn chế sự khác biệt giữa các reviewer. Nếu đối tượng bị cắt quá nhiều khiến không thể xác định chính xác loại vật thể hoặc ranh giới, cần escalation lên Lab Coach để được hướng dẫn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= [
  [446.0, 70.0],
  [445.0, 71.0],
  [444.0, 71.0],
  [443.0, 72.0],
  [442.0, 72.0]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ xác định vùng hình chữ nhật bao quanh vật thể, không thể hiện chính xác đường viền. Polygon mô tả đường biên thực tế của vật thể bằng nhiều điểm tọa độ, giúp xác định hình dạng, diện tích và những phần lồi lõm hoặc không đều của vật thể chính xác hơn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id là mã định danh riêng cho từng đối tượng cụ thể trong ảnh, giúp phân biệt các cá thể khác nhau dù chúng cùng một class.
- Không phải class ID vì class ID dùng để xác định loại đối tượng, ví dụ person, bowl.
- Không phải tracking ID vì tracking ID dùng để theo dõi cùng một đối tượng qua nhiều khung hình video.
Ví dụ: kitchen-001 xác định một người cụ thể trong ảnh kitchen, không có nghĩa là mã phân loại người hoặc mã theo dõi người qua video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ polygon sát theo đường viền thực tế của vật thể, chỉ bao gồm phần nhìn thấy được, không lấy thêm vùng nền hoặc vật thể bên cạnh. Khi vật thể bị che khuất, chỉ vẽ phần có thể quan sát được.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Chọn khu vực các chiếc bát (bowl) nằm sát và chồng lên nhau ở phía dưới bên trái ảnh, trên mặt bàn màu xanh. Ranh giới giữa một số chiếc bát khó xác định rõ do chúng tiếp xúc và che khuất nhau.
Tôi chọn escalation vì không thể xác định chắc chắn toàn bộ đường biên của từng chiếc bát. Cần chuyển trường hợp này cho người kiểm duyệt hoặc quản lý để thống nhất cách xử lý, tránh vẽ mask sai hoặc gộp nhiều vật thể thành một.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn class cho toàn bộ ảnh (image-level label), ví dụ: traffic, kitchen. | Ảnh có thể chứa nhiều vật thể hoặc chủ đề, khó xác định chủ đề chính cần phân loại. | Dựa vào guideline để chọn một nhãn phù hợp nhất với nội dung chính của ảnh. | Kiểm tra nhãn có phản ánh đúng nội dung toàn ảnh và thống nhất với taxonomy, guideline không. |
| Phát hiện vật thể | Mỗi vật thể có một class và một bounding box (tọa độ [x1, y1, x2, y2]). | JSON kitchen có nhiều prediction cùng class bowl, với score như 0.499948, 0.464904; ảnh có các vật thể nằm sát hoặc chồng lên nhau, dễ nhầm box. | Gán đúng class, vẽ box bao quanh từng vật thể riêng biệt, không bỏ sót hoặc gộp các vật thể. | Kiểm tra class, số lượng vật thể, box có bao sát vật thể không, có box trùng hoặc bỏ sót không. |
| Instance segmentation | Mỗi vật thể có class, instance ID và polygon/mask gồm các pixel thuộc vật thể đó. | Ảnh kitchen có các chiếc bát nằm sát nhau ở mặt bàn; một số vật thể bị che khuất. JSON có polygon với số điểm khác nhau, ví dụ kitchen-001 có 348 điểm. | Vẽ polygon sát đường biên thực tế của từng vật thể, tách riêng từng instance và chỉ gán phần nhìn thấy được. | Kiểm tra đường biên mask, độ chính xác polygon, class và instance ID; phát hiện mask chồng lấn hoặc gộp nhầm vật thể. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tôi chỉ sử dụng dữ liệu đúng phạm vi bài lab, không tự ý sao chép, chia sẻ, tải lên các nền tảng bên ngoài hoặc tiết lộ thông tin cá nhân trong ảnh và dữ liệu.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý, không tự ý sử dụng hoặc chia sẻ dữ liệu đó và báo ngay cho giảng viên hướng dẫn hoặc người quản lý lab để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
