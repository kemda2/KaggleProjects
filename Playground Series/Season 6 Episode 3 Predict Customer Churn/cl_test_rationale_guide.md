# 🧠 Senior Data Analisti Neden Her Testi Yapar?
## Pipeline Section 26–81 — Test Gerekçe Rehberi

> Bu rehber her testin **neden var olduğunu**, **hangi soruya cevap verdiğini**,
> **ne zaman devreye girdiğini** ve **sonuca göre ne yapılacağını** açıklar.
> Sadece kurallar değil: arka plandaki düşünce yapısı.

---

## Testin Arkasındaki 6 Temel Soru

Bir senior analist veriyi ilk gördüğünde kafasında şu 6 soru vardır.
Her test bu sorulardan birine cevap verir:

| Soru | Testler |
|---|---|
| **1. Bu veri güvenilir mi?** | S28, S27, S44, S45, S70, S71, S72, S73, S79 |
| **2. Model gerçekten öğreniyor mu?** | S65, S66, S55, S75, S36, S80 |
| **3. Hangi feature'lar gerçekten önemli?** | S26, S31, S43, S68, S69, S67, S41, S40, S42 |
| **4. Train ve test aynı dünyada mı?** | S35, S62, S64, S33, S56, S57 |
| **5. Model nasıl davranır ve ne kadar güvenilir?** | S36, S37, S38, S54, S76, S77, S78 |
| **6. Skoru nasıl artırırım?** | S49, S50, S58, S59, S60, S61 |

---

## Kategori 1: Veri Güvenilirlik Testleri
### "Bu veriye güvenebilir miyim?"

---

### S26 — LEAKAGE DETECTION
**Soru:** Bu feature production'da var olacak bilgiyi mi taşıyor, yoksa hedef değişkeni dolaylı mı ölçüyor?

**Neden önemli:** Leak olan feature ile model mükemmel AUC alır ama gerçek dünyada bu feature olmaz. Model production'da çöker. Kaggle'da submission sonrası fark edilirse tüm çalışma boşa gider.

**Ne zaman kritik:** Her zaman, ilk test olarak yapılır. Hiç atlanmaz.

**Mekanizma:** Her feature tek başına 5-fold RF ile test edilir. Solo AUC > 0.95 → o feature tek başına hedefi tahmin edebiliyor → leak.

**Sonuca göre aksiyon:**
- Solo AUC > 0.95 → Derhal drop et. Hiç düşünme.
- Solo AUC 0.85–0.95 → Domain bilgisiyle kontrol et. Production'da bu bilgi var mı?
- Solo AUC < 0.85 → Normal. Devam et.

---

### S27 — LABEL KALİTESİ & TARGET NOISE
**Soru:** Aynı feature profiline sahip iki müşteri farklı target almış olabilir mi?

**Neden önemli:** Label noise modelin öğrenemeyeceği "gürültü" yaratır. Conflict rate %10 ise modelin teorik üst sınırı zaten düşük demektir. Bunu bilmeden AUC yorumlamak yanıltıcı.

**Ne zaman kritik:** Birden fazla veri kaynağı birleştirildiğinde, insan etiketlemesi varsa, Kaggle synthetic veride.

**Mekanizma:** İlk 6 feature'a göre gruplama yapılır. Aynı grupta farklı target → conflict.

**Sonuca göre aksiyon:**
- Conflict > %10 → sample_weight ile "güvenilmez" satırlara düşük ağırlık ver.
- Conflict %5–10 → label_smoothing=0.05 LightGBM'e ekle.
- Conflict < %5 → Normal. Devam et.

---

### S28 — BUSINESS LOGIC VALIDATION
**Soru:** Sayılar matematiksel ve domain olarak tutarlı mı?

**Neden önemli:** "tenure=0, TotalCharges=150" → müşteri 0 ay kalmış ama para ödemiş. Bu imkansız. Bu tür satırlar modeli yanıltır. Ancak drop etmek yerine flag eklemek daha doğru çünkü anomali kendisi bilgi taşır.

**Ne zaman kritik:** Finansal, telco, sağlık verisi gibi domain kuralları net olan veri setlerinde.

**Mekanizma:** TotalCharges ≈ tenure × MonthlyCharges olmalı. Ratio 0.5–2.0 dışı → anomali.

**Sonuca göre aksiyon:**
- Anomali var → `is_charges_anomaly` binary feature ekle. Model bunu öğrenir.
- Anomali yok → Normal. Flag gerekmez.

---

### S44 — EKSİK DEĞER MEKANİZMASI (MCAR/MAR/MNAR)
**Soru:** Neden eksik bu değer? Eksiklik kendisi bir bilgi mi taşıyor?

**Neden önemli:** Eksiklik mekanizmasını bilmeden imputation stratejisi seçmek kör uçuş yapmak gibidir. MNAR durumunda mean impute yapmak hem yanlış hem de bilgi kaybına neden olur.

**Ne zaman kritik:** Null oranı > %5 olan her feature için zorunlu.

**Mekanizma:**
- MCAR: Null olan satırların churn rate'i null olmayanlara benzer + diğer feature'larla korelasyon yok → tamamen rastgele eksiklik.
- MAR: Diğer feature'larla korelasyon var → o feature'lara göre impute et.
- MNAR: Null olan satırlarda churn rate belirgin farklı (gap > 10p) → eksikliğin kendisi hedefle ilişkili.

**Sonuca göre aksiyon:**
- MCAR → mean/median impute güvenli.
- MAR → KNNImputer veya IterativeImputer.
- MNAR → `is_null` binary flag ZORUNLU. Imputation üstüne yapılır ama tek başına yetmez.

---

### S45 — OUTLIER TİPİ SINIFLANDIRMASI
**Soru:** Bu aykırı değer bir ölçüm hatası mı, gerçek bir uç müşteri mi, yoksa kendi segmenti olan farklı bir grup mu?

**Neden önemli:** Hepsine aynı Winsorize uygulamak yanlış. Veri girişi hatası → drop edilebilir. Gerçek uç değer → Winsorize. Segment → flag ekle, model öğrensin.

**Ne zaman kritik:** Outlier tespit edildiğinde, strateji belirlemeden önce.

**Mekanizma:**
- Hata tipi: z-score > 5 VE domain olarak imkansız değer.
- Segment tipi: Isolation Forest anormal işaretliyor + churn rate normal gruptan > 15p farklı.
- Gerçek uç değer: IQR dışı ama domain mümkün.

**Sonuca göre aksiyon:**
- Hata → impute et veya drop et.
- Segment → `is_anomaly` flag ekle.
- Uç değer → 1st-99th percentile Winsorize.

---

### S70 — DUPLICATE FEATURE DETECTION
**Soru:** İki farklı isimli feature aslında aynı bilgiyi mi taşıyor?

**Neden önemli:** Duplicate feature'lar modelde iki kez oy verir gibi etkir. Tree modeller bunu kısmen handle eder ama yorumlanabilirliği düşürür ve gereksiz karmaşıklık yaratır.

**Ne zaman kritik:** Çok sayıda feature varsa (> 20), birden fazla veri kaynağı birleştirildiğinde.

**Mekanizma:** Tüm numeric çiftler için Pearson |r| hesaplanır. |r| > 0.99 veya exact match > %99 → duplicate.

**Sonuca göre aksiyon:**
- Duplicate çift → ikisinden birini drop et. Hangisini? Domain açısından daha anlamlı olanı tut.
- Near-duplicate (r > 0.95) → ratio veya difference feature yap (A/B veya A-B). İki feature'ın bilgisi birleşir.

---

### S71 — NEAR-CONSTANT FEATURE DETECTION
**Soru:** Bu feature neredeyse hiç değişmiyor mu? Modele gerçek bilgi katıyor mu?

**Neden önemli:** Tek değer > %95 olan feature'dan model anlamlı bir şey öğrenemez. Ama gürültü ekler ve tree bölünmelerini boşa harcar.

**Ne zaman kritik:** Her zaman, özellikle imbalanced kategorik feature'larda.

**Eşik neden %95?** Mevcut pipeline > %99.5 kesiyordu — bu çok toleranslı. Bir feature'ın %96'sı aynı değerse, modelin bundan öğrenebileceği çok az şey var.

**Sonuca göre aksiyon:**
- Top value > %99 → Drop et.
- Top value %95–99 → İzle. Domain değeri varsa tut, yoksa drop et.

---

### S72 — STRING ANOMALY DETECTION
**Soru:** "Male", "male", "MALE" ayrı kategori olarak mı işleniyor?

**Neden önemli:** Yazım tutarsızlığı kategori sayısını şişirir, rare category sorununa neden olur ve OOD'a yol açar. Özellikle original Telco dataset ile Kaggle synthetic dataset birleştirildiğinde kritik.

**Ne zaman kritik:** Ham veri yüklendiğinde, birden fazla kaynak birleştirildiğinde.

**Mekanizma:** Original nunique ile lowercase+strip sonrası nunique karşılaştırılır. Azalmışsa sorun var.

**Sonuca göre aksiyon:**
- Collapse var → `.str.strip().str.lower()` uygula, sonra rare category analizini tekrar çalıştır.
- Collapse yok → Normal. Devam et.

---

### S73 — CROSS-FEATURE CONSISTENCY (Domain Kuralları)
**Soru:** "İnternet servisi yok ama StreamingTV = Yes" gibi imkansız kombinasyonlar var mı?

**Neden önemli:** Bu tür satırlar veri toplama hatasını işaret eder. Model bu tutarsızlıktan yanlış şeyler öğrenebilir. Aynı zamanda bu violation'ların kendisi churn ile ilişkili olabilir.

**Ne zaman kritik:** Domain kuralları net olan veri setlerinde. Telco'da yüksek öncelik.

**Sonuca göre aksiyon:**
- Violation < %5 → `is_domain_violation` flag ekle.
- Violation > %5 → Veri kaynağını sorgula. Hangi aşamada bozulmuş?

---

### S79 — BENFORD'S LAW
**Soru:** Sayısal veriler doğal mı görünüyor yoksa yapay/manipüle mi?

**Neden önemli:** Gerçek dünya sayıları Benford dağılımına uyar. Uymazsa: sentetik oluşturma, yuvarlama, veya manipülasyon olabilir. Kaggle'da synthetic verinin limitlerini anlamak için önemli.

**Ne zaman kritik:** Finansal veri, fraud tespiti, veri kalitesi şüphelenildiğinde.

**Sonuca göre aksiyon:**
- MAD > 0.015 → Veri doğal değil. Encoding stratejisini ve feature engineering'i buna göre ayarla.
- MAD < 0.006 → Doğal görünüyor. Güvenli.

---

## Kategori 2: Model Sağlığı Testleri
### "Model gerçekten öğreniyor mu, yoksa şans mı?"

---

### S65 — DUMMY CLASSIFIER BASELINE
**Soru:** Model random bir sınıflandırıcıyı ne kadar geçiyor?

**Neden önemli:** Yüksek AUC görünce "iyi model" demek yanlış. %26 churn rate olan veri setinde "herkesi No churner de" stratejisi %74 accuracy verir. Model bu baseline'ı ne kadar geçiyor?

**Ne zaman kritik:** İmbalanced veri setlerinde, düşük AUC gözlemlendiğinde, model ilk defa eğitildiğinde.

**Mekanizma:** Stratified, most_frequent ve prior strategy'li DummyClassifier'lar 5-fold ile değerlendirilir.

**Sonuca göre aksiyon:**
- Lift > 0.15 → Model gerçekten öğreniyor. Devam et.
- Lift 0.05–0.15 → Kabul edilebilir ama FE ile artır.
- Lift < 0.05 → Ciddi sorun. Leak kontrolü, imbalance stratejisi, ya da tamamen yeniden düşün.

---

### S66 — PERMUTATION TEST (Global Anlamlılık)
**Soru:** Modelin tüm öğrenimi şans eseri olabilir mi?

**Neden önemli:** DummyClassifier modelin "bir şeyler yaptığını" gösterir ama bu öğrenimin anlamlı olup olmadığını söylemez. Permutation test hedefi karıştırınca ne olduğunu ölçer. Gerçek AUC bu dağılımın dışında değilse → model şans üstünde değil.

**Ne zaman kritik:** Küçük veri setlerinde (< 5000 satır), şüpheli yüksek AUC'de, az özellik varken.

**Mekanizma:** Target 50 kez karıştırılır, her seferinde model eğitilir. p-value = kaçında gerçek AUC'den yüksek AUC elde edildi.

**Sonuca göre aksiyon:**
- p < 0.01 → Model kesinlikle anlamlı. Öğrenme gerçek.
- p < 0.05 → Anlamlı.
- p > 0.05 → Model şans üstünde değil. Ciddi sorun: feature kalitesi, veri boyutu.

---

### S55 — NESTED CV (Bias-Variance & Optimism)
**Soru:** Raporladığım AUC gerçek mi, yoksa hyperparameter tuning benim CV'mi de "overfit" etti mi?

**Neden önemli:** Standard 5-fold CV ile hem model seçip hem değerlendirirsen "optimistic bias" oluşur. Nested CV bunu engeller: outer loop gerçek performansı, inner loop hyperparameter seçimi yapar.

**Ne zaman kritik:** Grid search veya Optuna ile hyperparameter tuning yapıldıysa. Kaggle'da public LB'ye göre model seçimi yapıldıysa.

**Mekanizma:** Outer 3-fold = gerçek performans. Inner (şu an basitleştirilmiş, n_estimators sabit) = model eğitimi.

**Sonuca göre aksiyon:**
- Optimism > 0.01 → Standard CV skorun güvenilmez. Nested CV ile raporla.
- Optimism < 0.005 → Standard CV güvenilir. Devam et.

---

### S36 — CALİBRATION
**Soru:** Model "Bu müşterinin churn ihtimali %70" dediğinde gerçekten %70 mi?

**Neden önemli:** Calibration kötüyse threshold analizi anlamsız olur. "50% üzerindeyse churn tahmin et" dediğimizde model 0.3'ü "50%" olarak tahmin edebilir. Kötü kalibrasyon cost-sensitive analizini de bozar.

**Ne zaman kritik:** Threshold kullanılacaksa, olasılıklar iş kararlarına giriyorsa, cost matrix analizi yapılacaksa.

**Mekanizma:** OOF tahminleri 10 bin'e bölünür, her bin'in gerçek oranıyla karşılaştırılır.

**Sonuca göre aksiyon:**
- Calibration curve diyagonale yakın + Brier < 0.10 → İyi. Threshold güvenilir.
- Curve diyagonalden uzak → Isotonic regression veya Platt scaling uygula.

---

### S75 — SEED STABİLİTESİ
**Soru:** Farklı random_state ile AUC'ler ne kadar farklı? Model güvenilir mi?

**Neden önemli:** Seed değişince AUC 0.01 değişiyorsa modelin öğrenmesi stabil değil demektir. Bu şüpheli. Final submission için hangi seed'i seçeceğini bilmek de önemli.

**Ne zaman kritik:** Final model seçiminden önce. Kaggle'da seed fishing şüphelenildiğinde.

**Mekanizma:** 5 farklı seed ile 5-fold CV çalıştırılır. AUC std hesaplanır.

**Sonuca göre aksiyon:**
- Std > 0.005 → RepeatedStratifiedKFold kullan veya seed ortalaması al.
- Std < 0.002 → Stabil. Tek seed yeterli.

---

### S80 — McNEMAR'S TEST
**Soru:** "FE sonrası model daha iyi" diyorum — bu fark istatistiksel olarak gerçek mi?

**Neden önemli:** AUC 0.001 artabilir ama bu şans eseri olabilir. McNemar her iki modelin hangi örneklerde doğru/yanlış tahmin ettiğini karşılaştırır. "Daha iyi" olduğunu kanıtlar.

**Ne zaman kritik:** FE öncesi-sonrası karşılaştırmasında, iki model arasında seçim yaparken.

**Mekanizma:** b = full model doğru ama simple model yanlış; c = full model yanlış ama simple model doğru. Chi-square ile test edilir.

**Sonuca göre aksiyon:**
- p < 0.05 → Fark gerçek. Full model kullan.
- p > 0.05 → Fark şans eseri. Daha basit modelle gidebilirsin.

---

## Kategori 3: Feature Önceliklendirme Testleri
### "Hangi feature'lara zaman harcamalıyım?"

---

### S31 — SHAP & INTERACTION
**Soru:** Model hangi feature'a gerçekten bakıyor? Hangi feature çiftleri birlikte etki yapıyor?

**Neden önemli:** Feature importance'lar (gain, split) bias'lıdır. SHAP model-agnostic, her tahmin için feature katkısını gösterir. Interaction: A ve B birlikte bakıldığında tek tek bakılandan daha fazla bilgi var mı?

**Ne zaman kritik:** FE öncelik listesi oluştururken. Interaction feature yaratmadan önce.

---

### S43 — EFFECT SIZE ÖZETI
**Soru:** İstatistiksel anlamlı olan fark pratik olarak da önemli mi?

**Neden önemli:** 50.000 satır veri setinde 0.001 fark bile p < 0.001 verir. Ama bu fark anlamlı mı? Cohen d = 0.05 → Hayır. Büyük veri setlerinde p-value'ya körce güvenmek yanlış karar verilmesine neden olur.

**Ne zaman kritik:** Büyük veri setlerinde (> 10k). Her istatistiksel test sonrasında.

**Mekanizma:**
- Numeric: Cohen d = (mean_A - mean_B) / pooled_std
- Kategorik: Cramér V = sqrt(chi2 / (n * min_dim))

**Sonuca göre aksiyon:**
- Cohen d > 0.8 veya Cramér V > 0.5 → Büyük etki. FE'de ilk bu feature.
- Cohen d < 0.2 veya Cramér V < 0.1 → Pratik etkisi yok. Drop et.

---

### S67 — VIF (MULTICOLLINEARITY)
**Soru:** Bu feature başka bir feature'ın doğrusal kombinasyonu mu?

**Neden önemli:** VIF yüksek feature'lar gereksiz yere model kapasitesi harcar ve yorumlanabilirliği düşürür. Ayrıca, bir feature'ı drop ettiğinde başka bir feature'ın VIF'i dramatik düşüyorsa → ikisi arasında gizli ilişki var.

**Ne zaman kritik:** Çok sayıda numeric feature varsa. Ratio/difference FE yapmadan önce.

**Sonuca göre aksiyon:**
- VIF > 10 → İkisinin ratio veya farkını al (A/B veya A-B). Birini drop et.
- VIF > 5 → İzle.
- VIF < 5 → Güvenli.

---

### S68 — MI vs PEARSON TUTARSIZLIĞI
**Soru:** Bu feature'ın target ile ilişkisi doğrusal mı, non-linear mı?

**Neden önemli:** Pearson sadece doğrusal ilişkiyi ölçer. MI her türlü ilişkiyi ölçer. Pearson düşük ama MI yüksekse → feature önemli ama doğrusal değil. Direkt kullanmak yerine binning, log transform veya polynomial feature gerekli.

**Ne zaman kritik:** Feature seçimi yaparken, transform stratejisi belirlerken.

**Mekanizma:** MI normalize edilir (0–1). Pearson |r| ile karşılaştırılır. Gap > 0.2 → non-linear.

**Sonuca göre aksiyon:**
- Non-linear → PDP ile threshold bul, o noktada binary feature oluştur.
- Monoton non-linear → log1p veya sqrt transform.
- Linear → Direkt kullan.
- Weak (ikisi de düşük) → Drop et.

---

### S69 — SPEARMAN vs PEARSON
**Soru:** İlişki monoton ama eğri mi?

**Neden önemli:** Spearman sıralama korelasyonu ölçer — monoton ama non-linear ilişkileri yakalar. Pearson ölçmez. Gap > 0.10 → log1p transform işe yarar. Bu Section 68'i tamamlar.

**Mekanizma:** Her numeric feature için Spearman ve Pearson hesaplanır. Fark > 0.10 → monoton non-linear.

---

### S40 — LEVENE & MANN-WHITNEY U
**Soru:** Churn olan ve olmayan müşteriler arasında bu numeric feature gerçekten farklı mı?

**Neden önemli:** KDE ve KS testi grafik odaklıdır. Mann-Whitney resmi istatistiksel test verir. Levene varyans homojenliğini test eder — eğer varyanslar farklıysa hangi testin uygun olduğunu belirler.

**Sonuca göre aksiyon:**
- MWU p < 0.05 + Cohen d > 0.5 → Bu feature FE'de öncelikli.
- MWU p < 0.05 + Cohen d < 0.2 → İstatistiksel ama pratik değil. Büyük veri etkisi.

---

### S41 — KRUSKAL-WALLIS
**Soru:** Bu kategorik feature'ın farklı değerleri arasında churn rate anlamlı farklı mı?

**Neden önemli:** Chi-square sadece dağılım farkını test eder. Kruskal-Wallis "bu kategori grupları gerçekten farklı target'a sahip mi?" sorusunu yanıtlar. Eta-squared effect size da verir.

**Sonuca göre aksiyon:**
- p < 0.05 + Eta² > 0.14 → Target encoding için en güçlü aday.
- p >= 0.05 → Drop et.

---

### S42 — BONFERRONI/FDR CORRECTION
**Soru:** Bu p-value'lar arasında şans eseri anlamlı görünen var mı?

**Neden önemli:** 50 feature test edildiğinde, şans eseri 2–3 tanesi p < 0.05 çıkar. Bu false positive'dir. Bonferroni ve BH-FDR bu şişmeyi düzeltir.

**Sonuca göre aksiyon:**
- Bonferroni sonrası anlamlı → Kesinlikle önemli. Güvenle kullan.
- Sadece BH'de anlamlı → Dikkatli kullan. Effect size kontrol et.
- Hiçbiri kalmadı → Feature'lar gerçekten güçsüz. Agresif FE gerekli.

---

## Kategori 4: Drift ve Dağılım Testleri
### "Model test setinde de çalışacak mı?"

---

### S35 — PSI (POPULATION STABILITY INDEX)
**Soru:** Train'den test'e feature dağılımı ne kadar kaydı?

**Neden önemli:** KS test istatistiksel anlamlılık verir ama pratik anlam vermez. PSI "prodüksiyon monitoring standardı"dır — bankacılık ve telco'da yasal zorunluluk bile olabilir. PSI > 0.25 → model bu feature'dan öğrendiği şey test'te geçersiz.

**Ne zaman kritik:** Kaggle test seti ile her Kaggle yarışmasında. Production'da monthly monitoring.

**Sonuca göre aksiyon:**
- PSI > 0.25 → Ciddi. Drop veya RobustScaler/QuantileTransformer ile normalize et.
- PSI 0.10–0.25 → İzle. Normalize et.
- PSI < 0.10 → Güvenli.

---

### S64 — ADVERSARIAL VALİDATION
**Soru:** Train ve test setini ayırt edebilir miyiz? AUC = 0.50 ideal, 0.70+ ciddi sorun.

**Neden önemli:** PSI tek feature'ı ölçer. Adversarial validation tüm feature kombinasyonunun train/test'te ne kadar farklı olduğunu ölçer. Kaggle'da neredeyse standart hale gelmiştir. Ciddi shift varsa model test'te düşük puan alır.

**Mekanizma:** Train'e label=0, test'e label=1 ver. Bu "yeni hedef"i sınıflandır. AUC = ne kadar iyi ayırt edilebilir.

**Hangi feature'lar shift yapıyor:** Adversarial modelin importance'ı shift yapan feature'ları gösterir.

**Sonuca göre aksiyon:**
- AUC > 0.80 → Ciddi shift. Shift yapan feature'ları drop et veya target encode et.
- AUC 0.65–0.70 → Küçük shift. PSI izle.
- AUC < 0.60 → İdeal. Shift yok.

---

### S56 — OOF vs TEST DAĞILIMI
**Soru:** Modelin train'de öğrendiği tahmin dağılımı, test'te ürettiğiyle aynı mı?

**Neden önemli:** OOF tahminlerin dağılımı ile test tahminlerin dağılımı farklıysa → seçtiğin threshold OOF'a göre optimize edilmiş, test'te yanlış olabilir. Ayrıca covariate shift'in "tahmin düzeyindeki" yansımasıdır.

**Mekanizma:** KS testi ile OOF ve test tahmin dağılımları karşılaştırılır.

**Sonuca göre aksiyon:**
- KS > 0.10 → Rank transformation uygula. Ham olasılıklar yerine rank kullan.
- KS < 0.05 → Güvenli. Threshold doğrudan OOF'tan alınabilir.

---

### S57 — SATIR SIRASI BAĞIMLILIĞI
**Soru:** Veri satırları birbirinden bağımsız mı, yoksa ardışık satırlar korelasyonlu mu?

**Neden önemli:** CV shuffle=True varsayımla çalışır. Ama veri zaman sıralıysa bu shuffle anlamsız — aynı "zaman diliminden" satırlar hem train hem val'da olabilir. Bu train/val sızdırması yaratır.

**Sonuca göre aksiyon:**
- Autocorr > 0.2 → TimeSeriesSplit ile standart CV'yi karşılaştır.
- Autocorr < 0.05 → Shuffle güvenli. StratifiedKFold yeterli.

---

### S62 — CHI-SQUARE GOODNESS OF FIT
**Soru:** Kategorik feature'ların train/test dağılımı istatistiksel olarak aynı mı?

**Neden önemli:** PSI numeric'leri, Chi-square kategorikleri test eder. Kategorik drift → model o kategoriden yanlış genelleme yapar. Label encoding drift'i büyütür çünkü sayısal sıra anlamı değişir.

**Sonuca göre aksiyon:**
- p < 0.05 → Drift var. Target encoding kullan (label encoding değil).
- p >= 0.05 → Güvenli. Label encoding kullanılabilir.

---

### S33 — TEST OOD
**Soru:** Test'te train'de hiç görülmemiş değerler var mı?

**Neden önemli:** Model görmediği bir değerden extrapolate eder — bu güvenilmez. Kategorik OOD özellikle tehlikeli çünkü label encoder -1 verir ve model bu -1'i "öğrenilmiş bir kategori" olarak görür.

**Sonuca göre aksiyon:**
- Kategorik OOD → Unknown token ekle veya target encoding.
- Numeric OOD → Winsorize (train range'a clip).

---

## Kategori 5: Model Davranış Testleri
### "Model nasıl davranıyor, nerede yanılıyor?"

---

### S37 — THRESHOLD & COST-SENSITIVE
**Soru:** 0.5 gerçekten doğru threshold mı? Hangi threshold business'a en faydalı?

**Neden önemli:** Telco churn'de kaçırılan bir churner (FN) gereksiz müdahaleden (FP) genellikle 3–10x daha maliyetlidir. 0.5 threshold bu asimetriyi görmezden gelir. Cost-optimal threshold farklı olabilir.

**Ne zaman kritik:** Her zaman, model production'a gidecekse zorunlu.

**Mekanizma:** 200 farklı threshold için FN×fn_cost + FP×fp_cost hesaplanır. Minimum maliyet → optimal threshold.

**Sonuca göre aksiyon:**
- Optimal threshold 0.5'ten > 0.1 uzak → Default 0.5 kullanma. Business'a yanlış sonuç verir.
- Optimal threshold ≈ 0.5 → Default güvenli.

---

### S38 — LIFT & GAIN & DECILE
**Soru:** "Top %20 riski hedeflersek gerçek churner'ların kaçını yakalıyoruz?"

**Neden önemli:** Satış ve müşteri deneyimi ekipleri AUC bilmez. "Lift 2.5x" → "Rastgele seçime göre 2.5 kat daha iyi churner yakalıyoruz" diye anlatılabilir. Bu metrik model değerini business'a satmanın aracıdır.

**Sonuca göre aksiyon:**
- Lift < 1.5 → Model iş değeri zayıf. FE gerekli.
- Decile monoton değil → Model tutarsız. Threshold veya FE problemi.

---

### S54 — LEARNING CURVE
**Soru:** Modele daha fazla veri verirsek AUC artmaya devam eder mi?

**Neden önemli:** Augmentation kararı verilmeden önce öğrenme eğrisi görülmeli. Val AUC hâlâ yükseliyorsa → augmentation yardımcı olur. Platoya ulaşmışsa → daha iyi feature gerekiyor, veri değil.

**Sonuca göre aksiyon:**
- Val AUC hâlâ yükseliyor → Playground Series original data'yı concat et.
- Platoda ama iyi gap yok → Daha güçlü FE.
- Train >> Val → Overfitting. Regularizasyon.

---

### S76 — SHAP DEPENDENCE PLOTS
**Soru:** Bir feature'ın değeri arttıkça SHAP etkisi nasıl değişiyor? Nerede kırılma var?

**Neden önemli:** SHAP importance "ne kadar önemli" söyler. Dependence "nasıl davranıyor" söyler. S-eğrisi veya keskin kırılma → o noktada binary threshold feature oluşturmak gerekir.

**Ne zaman kritik:** PDP'yi tamamlar. PDP ortalama etkiyi, SHAP dependence bireysel etkileri gösterir.

---

### S77 — FEATURE STABİLİTESİ ACROSS FOLDS
**Soru:** Bir feature fold'dan fold'a tutarlı bir öneme sahip mi?

**Neden önemli:** SHAP importance fold'dan fold'a çok değişen feature → o feature modeli fold'a göre farklı kullanıyor. Bu instabilitenin nedenini anlamak FE yönünü belirler.

**Sonuca göre aksiyon:**
- CV (std/mean) > 0.5 → Instabil. Regularizasyon veya drop.
- CV < 0.3 + yüksek importance → Güvenilir feature. FE'de öncelikli kullan.

---

### S78 — HARD SAMPLE SHAP FORCE
**Soru:** Model neden bu satırları sürekli yanlış tahmin ediyor?

**Neden önemli:** Hard sample'lar modelin kör noktasını gösterir. Eğer tüm hard sample'lar tenure=0–3 civarındaysa → yeni müşteri segmenti için özel FE gerekiyor. Rastgele FE değil, hedefli FE.

**Sonuca göre aksiyon:**
- Hard sample'larda baskın feature bulundu → O feature için özel transform veya interaction.

---

## Kategori 6: Skor Artırma Testleri
### "Mevcut modeli nasıl daha iyiye taşırım?"

---

### S29 — COHORT DRIFT
**Soru:** Tenure grubuna göre churn rate ne kadar farklı?

**Neden önemli:** Yeni müşteriler (tenure < 12) ile eski müşteriler (tenure > 48) tamamen farklı profillere sahip. Eğer model ikisini aynı feature space'de değerlendirirse yanlış genelleme yapar. Cohort spread > 20p → segmentasyon feature şart.

---

### S30 — STRATİFİED KDE
**Soru:** Bu numeric feature Churn=Yes ve No gruplarını ne kadar ayırt ediyor?

**Neden önemli:** KS > 0.3 → güçlü ayrıştırıcı. Bu feature'lar FE'de öncelikli. KS < 0.05 → feature neredeyse işe yaramıyor.

---

### S32 — HARD SAMPLE ANALİZİ
**Soru:** 5 fold'un 4'ünde yanlış tahmin edilen satırlar hangileri?

**Neden önemli:** Model bu satırları sistematik olarak yanlış yapıyor. Bunların numeric ortalaması "normal" satırlardan çok farklıysa → o feature'lar için özel FE gerekiyor.

---

### S49 — TARGET ENCODING LEAKAGE SİMÜLASYONU
**Soru:** Target encoding pipeline'ım gerçekten CV-safe mi?

**Neden önemli:** Naif target encoding (tüm train mean'ini kullanmak) CV loop'unu kirletir. AUC şişer. Gerçek test performansı daha kötü olur. Bu simülasyon leakage miktarını ölçer.

---

### S50 — OPTİMAL BİNNİNG
**Soru:** pd.qcut mu daha iyi IV verir, yoksa IV-optimize binning mi?

**Neden önemli:** Binning stratejisi doğrudan FE kalitesini etkiler. Optimal binning WoE monotonluğunu korurken IV'yi maksimize eder.

---

### S58 — INTERACTION BOOTSTRAP CI
**Soru:** A×B çarpım feature'ı gerçekten AUC artırıyor mu, yoksa şans mı?

**Neden önemli:** Bootstrap CI ile güven aralığı hesaplamadan "bu interaction işe yarıyor" demek yanlış. CI alt sınırı > 0 olmalı.

---

### S59 — ENCODING STRATEJİSİ KARŞILAŞTIRMASI
**Soru:** Bu kategorik feature için hangi encoding en yüksek AUC veriyor?

**Neden önemli:** Varsayılan label encoding her zaman en iyi değildir. Yüksek kardinaliteli feature'larda target encoding çok daha güçlü olabilir. Bunu ölçmeden seçim yapma.

---

### S60 — POST-PROCESSING
**Soru:** Rank transform veya power transform AUC artırıyor mu?

**Neden önemli:** Modeli değiştirmeden sadece çıktı transformasyonu ile 0.001–0.003 AUC kazanılabilir. Az eforla yüksek kazanç.

---

### S61 — SUBMISSION KORELASYONU
**Soru:** Bu iki model tahminleri ne kadar korelasyonlu? Ensemble yapmaya değer mi?

**Neden önemli:** Korelasyon yüksek (> 0.98) → aynı şeyi öğrenmişler. Blend değeri düşük. Korelasyon düşük (< 0.90) → farklı bilgi öğrenmişler. Blend +0.002–0.005 kazandırabilir.

---

## Tüm Testlerin Özet Haritası

```
VERİ GELDİ
    │
    ├─► S72 String anomaly → normalize
    ├─► S71 Near-constant → drop
    ├─► S70 Duplicate → drop/combine
    ├─► S73 Domain rules → flag
    ├─► S28 Business logic → flag
    ├─► S44 MCAR/MAR/MNAR → imputation stratejisi
    ├─► S45 Outlier tipi → per-type strateji
    └─► S79 Benford → veri güveni

MODEL EĞİTİLDİ (Baseline)
    │
    ├─► S65 Dummy baseline → gerçekten öğreniyor mu?
    ├─► S66 Permutation test → anlamlı mı?
    ├─► S26 Leakage detection → leak var mı?
    ├─► S27 Label quality → noise seviyesi?
    ├─► S55 Nested CV → AUC güvenilir mi?
    ├─► S36 Calibration → olasılıklar doğru mu?
    ├─► S75 Seed stability → tekrarlanabilir mi?
    └─► S74 Stratification → CV fold'ları dengeli mi?

FEATURE ANALİZİ
    │
    ├─► S31 SHAP → hangi feature önemli?
    ├─► S43 Effect size → pratik önem?
    ├─► S67 VIF → multicollinearity?
    ├─► S68 MI vs Pearson → non-linear?
    ├─► S69 Spearman vs Pearson → monoton non-linear?
    ├─► S40 Mann-Whitney → grup farkı?
    ├─► S41 Kruskal-Wallis → kategorik etki?
    └─► S42 Bonferroni → false positive düzeltme

DRIFT & DAĞILIM
    │
    ├─► S64 Adversarial validation → genel drift?
    ├─► S35 PSI → numeric feature drift?
    ├─► S62 Chi-square → kategorik drift?
    ├─► S33 OOD → görülmemiş değer?
    ├─► S56 OOF vs test dist → tahmin drift?
    └─► S57 Autocorrelation → satır bağımlılığı?

MODEL DAVRANIŞI
    │
    ├─► S37 Threshold + cost matrix → doğru eşik?
    ├─► S38 Lift + decile → iş değeri?
    ├─► S54 Learning curve → daha veri lazım mı?
    ├─► S76 SHAP dependence → threshold noktaları?
    ├─► S77 Feature stability → stabil mi?
    └─► S78 Hard sample SHAP → neden yanılıyor?

FE & OPTİMİZASYON
    │
    ├─► S29 Cohort drift → segmentasyon feature?
    ├─► S30 KDE → güçlü ayrıştırıcı?
    ├─► S32 Hard samples → hedefli FE?
    ├─► S47 Null patterns → null feature?
    ├─► S48 Rare categories → birleştir/flag?
    ├─► S49 TE leakage sim → KFold TE gerekli mi?
    ├─► S50 Optimal binning → qcut mu, optimal mi?
    ├─► S51 PDP → threshold yerleri?
    ├─► S52 Anomaly score → feature olarak?
    ├─► S53 Quantile rate → ani sıçrama?
    ├─► S58 Interaction CI → interaction anlamlı mı?
    ├─► S59 Encoding karşılaştırma → en iyi encoding?
    ├─► S60 Post-processing → rank/power transform?
    ├─► S61 Submission korelasyon → ensemble değeri?
    └─► S80 McNemar → fark istatistiksel mi?

HAZIRLIK SKORU (S81)
    └─► FE'ye geç veya sorunları gider
```

---

## Hızlı Başvuru: Hangi Testi Ne Zaman Atlarım?

| Durum | Atlayabileceğin Testler |
|---|---|
| Veri > 100k satır | S66 Permutation (yavaş), S79 Benford |
| Null yok | S44 MCAR/MAR/MNAR, S47 Null patterns |
| Tüm feature'lar numeric | S72 String anomaly, S41 KW, S59 Encoding |
| Tüm feature'lar kategorik | S67 VIF, S40 MWU, S69 Spearman, S46 PCA |
| Kaggle (skor odaklı) | S73 Domain rules, S79 Benford |
| Production (risk odaklı) | S60 Post-processing, S61 Submission corr |
| **Hiç atlanmaz** | **S26 Leak, S65 Dummy, S64 Adversarial, S36 Calibration, S37 Threshold** |
