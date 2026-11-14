# Báo cáo cá nhân — Day 6 · Data Ingestion Engineering

- **Họ và tên:** Nguyễn Tiến Đạt
- **MSSV:** 2A202603452
- **Lớp:** L3A

---

## 1. Dữ liệu và pipeline

### Chủ đề, nguồn và quyền
- **Chủ đề:** Sách giáo khoa *Prealgebra 2e* (OpenStax), chương **1 "Whole Numbers"** và **2 "The Language of Algebra"**.
- **Nguồn:** https://openstax.org/books/prealgebra-2e
- **PDF:** `https://assets.openstax.org/oscms-prodcms/media/documents/prealgebra-2e_-_WEB.pdf` (lưu tại `starter/data/prealgebra-2e.pdf`, 1074 trang PDF).
- **Giấy phép:** **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0) — chỉ dùng phi thương mại, phái sinh phải giữ cùng license. Bài lab này là mục đích giáo dục phi thương mại nên được phép.
- **Số trang:** chọn **163 trang nội dung** (trang in 7–169), vượt mức tối thiểu 30 trang. Quy tắc tính: dùng số trang in ở chân trang; loại bỏ phần đầu (bìa, copyright, mục lục, preface) và phần cuối (đáp án, index). **Trang in = chỉ số PDF − 9** (kiểm chứng qua footer trên nhiều trang). Toàn bộ 163 trang đều có text (0 trang trống).

### Chuẩn hóa, lỗi và giới hạn
- `normalize_text`: chuẩn hóa Unicode **NFKC**, thay soft-hyphen/non-breaking-space, thay ký tự không ánh xạ `U+FFFD` bằng khoảng trắng, gộp khoảng trắng (giữ dấu chấm câu để tách câu).
- **Giới hạn đã biết:** các ký hiệu toán (số trong công thức, hình) bị pypdf bỏ mất, nên một số định nghĩa trong thân chương mất số (vd "greater than 1" → "greater than"). Do đó các câu hỏi được viết ưu tiên các định nghĩa bằng lời, và evidence được trích **đúng** từ văn bản đã trích xuất (xác minh tự động bằng `scripts/verify_benchmark.py`, 20/20 hợp lệ). Không dùng OCR vì PDF là text gốc.

### Embedding model
- **Provider/model:** `intfloat/multilingual-e5-base` (Sentence Transformers, chạy local, miễn phí).
- **Số chiều:** 768.
- **Cách xử lý query/document:** query dùng prefix `"query: "`, document dùng `"passage: "` (quy ước E5 cho truy xuất đa ngữ). Vector được chuẩn hóa L2.
- **Lý do chọn:** ban đầu dùng `paraphrase-multilingual-MiniLM-L12-v2` (384d) theo gợi ý, nhưng thử nghiệm cho thấy truy xuất Việt→Anh rất yếu (trúng 2/5 trang). `multilingual-e5-base` trúng 4/5 và cho kết quả có ý nghĩa hơn, nên được chọn. Cả hai đều là model thật, không phải mock.

### Vector store
- **Store:** PostgreSQL 16 + **pgvector 0.8.7** (Docker Compose, cổng local `127.0.0.1:55433`), metric **cosine** (`<=>`) cho cả ba cấu hình. Lưu bền vững (volume), hỗ trợ vector search + metadata filter. Index HNSW `vector_cosine_ops`.
- **Exact vs approximate:** truy vấn xếp hạng theo cosine distance `<=>`; có index HNSW (ANN), nhưng ở quy mô ~550 vector kết quả gần như exact (độ xấp xỉ không đáng kể). Không dùng vector mock/ngẫu nhiên; nạp lại bằng `DELETE` + `INSERT` theo chiến lược nên không nhân đôi (đã kiểm tra: count == distinct chunk_id, A 519/519, B 550/550, C 550/550).

### Metadata
| Trường | Kiểu | Nguồn | Lý do |
|---|---|---|---|
| `document_id` | string | gán `prealgebra-2e` | truy vết |
| `source` | string | tên file | truy vết |
| `page` | int | số trang in | truy vết / gold |
| `chapter` | string | mục lục | **trường filter** (C) |
| `section` | string | mục lục | truy vết chi tiết hơn |

`chapter` có 2 giá trị phân biệt ("Whole Numbers", "The Language of Algebra") — đủ để test filter.

### Lệnh chạy & chi phí
```powershell
cd starter
docker compose -p day6_student_starter up -d --wait   # nếu DB chưa chạy
python -m src.run                                      # chạy toàn bộ pipeline + benchmark
python scripts/verify_benchmark.py                    # kiểm chứng evidence
```
- **Chi phí:** 0 (model local, không API trả phí).
- **Thời gian ingest (embedding chiếm phần lớn):** A ≈ 174s, B ≈ 136s, C ≈ 129s.
- **Phiên bản thư viện & thời điểm chạy:** xem `starter/results/processing_log.md` (sentence-transformers 6.3.0, torch 2.15.0, transformers 5.22.2, psycopg 3.3.8, pgvector 0.5.1, pypdf 6.21.4, numpy 2.5.4).

---

## 2. Cấu hình và benchmark

| Cấu hình | Chunk | Metadata / truy xuất |
|---|---|---|
| **A** | Kích thước cố định 500 ký tự, overlap 50 | source + page + chapter/section; **không lọc** |
| **B** | Theo câu (sentence-aware), giới hạn 500 ký tự | cùng metadata; **không lọc** |
| **C** | Giống B | **thêm filter theo `chapter`**, chỉ áp dụng khi câu hỏi có filter |

- **A–B:** chỉ đổi cách chunk (cửa sổ cố định vs theo câu). **B–C:** chỉ đổi lọc (chunk giống hệt).
- Cùng corpus, cùng model, cùng store/cosine, cùng 20 câu hỏi.

### Benchmark
- **20 câu hỏi** (15 dev / 5 test), viết **trước** khi chạy. Có đủ loại: trực tiếp, diễn đạt lại, điều kiện/ngoại lệ.
- **3 câu filter** (`chapter`), trong đó **1 câu test** (q20). Giá trị filter được ngữ cảnh câu hỏi cung cấp trước khi truy xuất, không chứa đáp án/gold source.
- Mỗi câu có `expected_answer`, `gold_source_id`, `gold_page`, `evidence_quote` đã **kiểm chứng nằm đúng trang nguồn** (20/20, tự động). Không có câu nào không có đáp án trong nguồn.
- **Tiêu chí chọn chiến lược (chốt trước khi xem test):** ưu tiên **Evidence Hit@3** (test), rồi **grounding**, rồi **coherence**, rồi **tài nguyên** (số chunk/thời gian).
- **Nguy cơ rò rỉ:** dev và test lấy từ cùng 2 chương nên có chủ đề chồng lấn (vd q13/q17/q20 đều liên quan số nguyên tố/hợp số). Đã tách dev/test **trước** khi chạy và không điều chỉnh chiến lược dựa trên kết quả test.

---

## 3. Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 6/15 | 8/15 | 519 | ~174s | 0 |
| A | test | 2/5 | 3/5 | | | |
| B | dev | 4/15 | 9/15 | 550 | ~136s | 0 |
| B | test | 2/5 | 4/5 | | | |
| C | dev | 4/15 | 9/15 | 550 | ~129s | 0 |
| C | test | 2/5 | 4/5 | | | |

| Strategy | Coherence TB /2 | Metadata đủ & đúng (n/N) | Filter tuân thủ (n/N) | Grounding (0/1/2) và TB /2 |
|---|---|---|---|---|
| A | **0.6** | 15/15 | N/A (không filter) | 0×0, 1×1, 4×2 → **1.8** |
| B | **1.6** | 15/15 | N/A | 0×0, 1×1, 4×2 → **1.8** |
| C | **1.6** | 15/15 | **3/3** | 0×0, 1×1, 4×2 → **1.8** |

Đường dẫn: `starter/results/summary.json`, `trace_A/B/C.json`, `manual_worksheet.json`, `manual_scores.json`.

---

## 4. Phân tích và kết luận

### So sánh A–B (chunking)
- **Hit@1 cao hơn ở A** (dev 6/15 vs 4/15) nhưng **Hit@3 cao hơn ở B** (dev 8→9, test 3→4). Cửa sổ cố định của A đôi khi gom cả câu trả lời vào một chunk top-1 nhờ overlap, nhưng cắt giữa câu nên evidence thường bị chia đôi → thua ở Hit@3. Chunk theo câu của B giữ trọn evidence nên Hit@3 ổn định hơn.
- **Coherence:** A = 0.6/2 vs B = 1.6/2 — khác biệt rõ do A cắt giữa từ ở **cả 10 vị trí** khảo sát, còn B kết thúc ở ranh giới câu.

### So sánh B–C (metadata filter)
- Filter hoạt động **đúng kỹ thuật**: 3/3 câu filter có top-3 tuân thủ `chapter`. Tuy nhiên kết quả **không đổi** so với B vì các chương ngữ nghĩa tách biệt — câu hỏi vốn đã truy xuất đúng chương. Đây là **hạn chế trung thực**: filter chỉ *đảm bảo phạm vi*, chưa *cải thiện xếp hạng* trên corpus này; giá trị thực của nó rõ hơn ở corpus có từ vựng trùng lặp giữa các phần.

### Hai ví dụ sai (kèm ID câu + nguồn)
1. **q19** ("Bội chung nhỏ nhất của 15 và 18 là bao nhiêu?" → 90, trang 159): B và C không đưa trang 159 vào top-3 (chỉ trả định nghĩa LCM ở trang 157 và bài tập 160/169) → thiếu đáp án cụ thể "90". Model map "LCM của 15 và 18" về khái niệm LCM chung chứ không về ví dụ cụ thể.
2. **q14** ("Trong chương Whole Numbers, chu vi được tính thế nào?" → trang 34): cả ba cấu hình đều trượt trang 34, trả về trang phép trừ (49, 64). Từ "chu vi" → "perimeter" map yếu, và câu trả lời "sum of the lengths of its sides" bị lẫn với nội dung phép cộng.

### Chiến lược chọn, đánh đổi, điều kiện áp dụng
- Chọn **B (và C = B + filter khi có yêu cầu lọc)** vì Hit@3 test cao nhất (4/5), coherence cao nhất (1.6), chi phí tương đương. A có Hit@1 nhỉnh hơn nhưng coherence và Hit@3 thấp hơn.
- **Đánh đổi:** B tạo nhiều chunk hơn (550 vs 519) và có thể sinh chunk quá ngắn (đã gộp mảnh vụn); C tăng chi phí lưu/hiểu metadata nhưng không cải thiện xếp hạng trên corpus này.

### Hạn chế
- Corpus chỉ 2 chương của 1 sách tiếng Anh; câu hỏi tiếng Việt → truy xuất **xuyên ngôn ngữ**, khó hơn đơn ngữ.
- Tập test nhỏ (5 câu), chỉ số có phương sai lớn; không suy rộng ra ngoài corpus.
- Evidence-matching tự động là khớp chuỗi con chính xác: evidence bị tách qua nhiều chunk sẽ bị tính là miss (vd q19 đối với A ở mức ký tự).
- Chấm thủ công do chính tác giả thực hiện → có nguy cơ thiên lệch; thời gian đo trên CPU local, không ổn định tuyệt đối.
- B và C cùng embedding được nhúng hai lần (lãng phí); có thể cache chung.

### Nếu làm lại / mở rộng
- Dùng model mạnh hơn (`multilingual-e5-large`) hoặc dịch corpus sang tiếng Việt để cải thiện cross-lingual.
- Thêm hybrid BM25 + dense để xử lý truy vấn chứa số cụ thể (vd "15 và 18").
- Tách gold page khi evidence lặp ở nhiều trang (q20 có định nghĩa ở cả trang 148 và 162).
- Chọn corpus có metadata filter thực sự phân tách (nhiều tài liệu/năm) để chứng minh giá trị filter rõ hơn.
