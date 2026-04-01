# Overview

Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Predict the irrigation need.

Start
6 hours ago
Close
a month to go
Evaluation

Submissions are evaluated on balanced accuracy between the predicted class and observed target.
Submission File

For each id in the test set, you must predict a class label (Low, Medium, High) for the Irrigation_Need target. The file should contain a header and have the following format:

id,Irrigation_Need
630000,Low
630001,High
630002,Low
etc.

Timeline

    Start Date - April 1, 2026
    Entry Deadline - Same as the Final Submission Deadline
    Team Merger Deadline - Same as the Final Submission Deadline
    Final Submission Deadline - April 30, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.
Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!
Prizes

    1st Place - Choice of Kaggle merchandise
    2nd Place - Choice of Kaggle merchandise
    3rd Place - Choice of Kaggle merchandise

Please note: In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team.

# Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Irrigation Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
Files

    train.csv - the training set, with Irrigation_Need as target
    test.csv - the test set, used to predict the category for Irrigation_Need
    sample_submission.csv - a sample submission file in the correct format





# Genel Bakış

2026 Kaggle Playground Serisi’ne hoş geldiniz! Önceki playground yarışmalarının ruhunu sürdürmeyi planlıyoruz. Topluluğumuzun makine öğrenimi becerilerini geliştirebilmesi için ilgi çekici ve erişilebilir veri setleri sunmaya devam edeceğiz ve her ay bir yarışma düzenlemeyi öngörüyoruz.

**Hedefiniz:** Sulama ihtiyacını tahmin etmek.

**Başlangıç:** 6 saat önce
**Bitiş:** Bitmesine 1 ay var

---

**Değerlendirme**

Gönderimler, tahmin edilen sınıf ile gerçek hedef arasındaki **balanced accuracy (dengeli doğruluk)** metriğine göre değerlendirilecektir.

---

**Gönderim Dosyası**

Test setindeki her bir id için, Irrigation_Need hedefi adına bir sınıf etiketi (Low, Medium, High) tahmin etmelisiniz. Dosya bir başlık içermeli ve aşağıdaki formatta olmalıdır:

```
id,Irrigation_Need
630000,Low
630001,High
630002,Low
...
```

---

**Zaman Çizelgesi**

* Başlangıç Tarihi – 1 Nisan 2026
* Katılım Son Tarihi – Final gönderim tarihi ile aynı
* Takım Birleşme Son Tarihi – Final gönderim tarihi ile aynı
* Final Gönderim Son Tarihi – 30 Nisan 2026

Aksi belirtilmedikçe tüm son tarihler ilgili günün UTC saatine göre 23:59’dur. Yarışma organizatörleri gerekli gördükleri takdirde zaman çizelgesini güncelleme hakkını saklı tutar.

---

**Tabular Playground Serisi Hakkında**

Tabular Playground Serisi’nin amacı, Kaggle topluluğuna makine öğrenimi ve veri biliminin farklı alanlarında becerilerini öğrenip geliştirebilecekleri, nispeten hafif ve çeşitli zorluklar sunmaktır. Her yarışma genellikle birkaç hafta sürer, ancak zorluğa bağlı olarak daha uzun veya kısa olabilir.

Bu yarışmalar genellikle gerçek dünya verilerinden türetilmiş sentetik (yapay) veri setleri kullanır. Katılımcılar bu sayede hızlıca farklı model ve özellik mühendisliği denemeleri yapabilir, görselleştirmeler oluşturabilir ve fikirlerini test edebilir.

---

**Sentetik Veri Setleri**

Playground yarışmalarında sentetik veri kullanımı, gerçek dünya verilerine benzer (isimlendirilmiş özelliklere sahip) veri sağlarken, test etiketlerinin herkese açık olmamasını garanti eder. Bu da daha ilgi çekici yarışmalar düzenlememize olanak tanır.

Sentetik veri üretiminde hâlâ bazı zorluklar olsa da, günümüzde kullanılan yöntemler önceye göre çok daha gelişmiştir ve veri setlerinde daha az yapay iz (artifact) bulunması hedeflenmektedir. Veri setleri hakkında geri bildirimlerinizi paylaşmanız, gelecekteki yarışmaları geliştirmemize yardımcı olacaktır.

---

**Ödüller**

* 1.’lik – Kaggle ürünlerinden seçim
* 2.’lik – Kaggle ürünlerinden seçim
* 3.’lük – Kaggle ürünlerinden seçim

**Not:** Yeni başlayanların katılımını teşvik etmek amacıyla, bu seri kapsamında Kaggle ödülleri kişi başına yalnızca bir kez verilecektir. Daha önce kazanmış bir kişi varsa, ödül bir sonraki takıma verilecektir.


# Veri Seti Açıklaması

Bu yarışma için kullanılan veri seti (hem eğitim hem de test), Sulama Tahmini (Irrigation Prediction) veri seti üzerinde eğitilmiş bir derin öğrenme modeli tarafından üretilmiştir. Özellik dağılımları orijinal veri setine oldukça benzer olsa da birebir aynı değildir.

Orijinal veri setini de bu yarışma kapsamında kullanabilirsiniz. Hem iki veri seti arasındaki farkları incelemek hem de orijinal veriyi eğitime dahil etmenin model performansını iyileştirip iyileştirmediğini görmek için bu yaklaşımı deneyebilirsiniz.

---

**Dosyalar**

* **train.csv** – Hedef değişkeni *Irrigation_Need* olan eğitim veri seti
* **test.csv** – *Irrigation_Need* kategorisini tahmin etmek için kullanılan test veri seti
* **sample_submission.csv** – Doğru formatta hazırlanmış örnek gönderim dosyası
