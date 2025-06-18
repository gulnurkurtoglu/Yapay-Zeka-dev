Moda Ürünleri Renk Analizi
Bu proje, moda ürün açıklamalarında geçen renk isimlerini analiz ederek, bu renklerin anlamsal olarak hangi CSS renklerine daha yakın olduğunu belirlemeyi amaçlamaktadır. Doğal dil işleme ve Word2Vec modelleme teknikleri kullanılarak, renk kelimeleriyle CSS renkleri arasında bağ kurulmuştur.

Proje Amacı
Moda sektöründe kullanılan ürün açıklamaları, çoğu zaman kullanıcıya görsel bir izlenim sunar. Bu açıklamalardaki renk isimlerini analiz edip, bilgisayarla anlaşılır hale getirmek, özellikle görsel eşleşme, otomatik etiketleme ve stil öneri sistemleri için önemlidir. Bu projede renk isimleri doğal dil bağlamında analiz edilmiştir.

Yöntem
Proje kapsamında öncelikle metin verileri temizlenmiş, stopword'ler çıkarılmıştır. Ardından bu verilerle bir Word2Vec modeli eğitilmiş ve belirlenen temel renk isimleri model üzerinden vektörleştirilmiştir. Her bir renk kelimesi, CSS renkleriyle karşılaştırılarak en yakın eşleşme bulunmuştur. Benzerlik ölçümü için cosine benzerliği kullanılmıştır.

Çıktılar
Analiz sonucunda, örneğin “kırmızı” kelimesinin hangi CSS rengini temsil ettiği gibi eşleşmeler elde edilmiştir. Elde edilen bu eşleşmeler sayesinde, metinsel renk ifadeleri sayısal olarak anlamlandırılmıştır. Sonuçlar sade bir liste olarak CSV dosyasına kaydedilmiş, görsel örnekler ise isteğe bağlı olarak görselleştirilmiştir.

Gereksinimler
Bu proje Python ile yazılmıştır. Gerekli kütüphaneler arasında pandas, nltk, gensim, matplotlib ve webcolors yer almaktadır.

Çalıştırma
Tüm analiz adımları Jupyter Notebook içerisinde adım adım ilerlemektedir. Veri seti ve notebook aynı dizinde yer almalıdır. İlk kez çalıştıranlar için gerekli olan NLTK verileri de indirilmektedir.

Sonuç
Bu proje ile doğal dilde geçen renk kelimeleri, yapay zekâ temelli bir modelle CSS renkleriyle eşleştirilmiştir. Moda, tasarım ve e-ticaret alanlarında kullanılabilecek daha büyük sistemlerin temelini oluşturabilecek bir adımdır.


