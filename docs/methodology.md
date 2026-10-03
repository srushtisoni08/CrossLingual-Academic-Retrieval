en -> en              0.994     0.878
en -> hi              0.951     0.761
en -> bn              0.920     0.728
en -> te              0.917     0.711
hi -> en              0.891     0.691
hi -> hi              0.973     0.832
hi -> bn              0.935     0.760
hi -> te              0.916     0.727
bn -> en              0.818     0.606
bn -> hi              0.911     0.726
bn -> bn              0.965     0.815
bn -> te              0.885     0.693
te -> en              0.782     0.579
te -> hi              0.896     0.702
te -> bn              0.883     0.690
te -> te              0.964     0.813




Model: intfloat/multilingual-e5-small   (top-1 cosine similarity; stub passages excluded)

== Corpus: en ==
  off-topic probes   min / median / max : 0.767 / 0.797 / 0.843
  real, same-lang    p5 / p25 / median  : 0.842 / 0.864 / 0.882
  real, cross-lang   p5 / p25 / median  : 0.770 / 0.788 / 0.804
  threshold = max probe + 0.005 = 0.848  ->  would wrongly flag 8.6% of same-lang and 94.1% of cross-lang real queries

== Corpus: hi ==
  off-topic probes   min / median / max : 0.781 / 0.802 / 0.819
  real, same-lang    p5 / p25 / median  : 0.841 / 0.865 / 0.882
  real, cross-lang   p5 / p25 / median  : 0.792 / 0.812 / 0.829
  threshold = max probe + 0.005 = 0.824  ->  would wrongly flag 1.2% of same-lang and 42.9% of cross-lang real queries

== Corpus: bn ==
  off-topic probes   min / median / max : 0.773 / 0.808 / 0.827
  real, same-lang    p5 / p25 / median  : 0.838 / 0.863 / 0.881
  real, cross-lang   p5 / p25 / median  : 0.790 / 0.815 / 0.832
  threshold = max probe + 0.005 = 0.832  ->  would wrongly flag 3.2% of same-lang and 49.0% of cross-lang real queries

== Corpus: te ==
  off-topic probes   min / median / max : 0.777 / 0.810 / 0.834
  real, same-lang    p5 / p25 / median  : 0.842 / 0.867 / 0.884
  real, cross-lang   p5 / p25 / median  : 0.791 / 0.814 / 0.832
  threshold = max probe + 0.005 = 0.839  ->  would wrongly flag 4.0% of same-lang and 60.1% of cross-lang real queries
(venv) PS D:\nlp\car\backend> 



Model: intfloat/multilingual-e5-small   (stub passages excluded; AUC 1.0 = perfect, 0.5 = useless)

== Corpus: en ==
  [same-lang]
    similarity  probes max 0.843 | real p5 0.842 median 0.882 | AUC 0.98
    top1-top10  probes max 0.027 | real p5 0.038 median 0.080 | AUC 0.99
  [cross-lang]
    similarity  probes max 0.822 | real p5 0.770 median 0.804 | AUC 0.62
    top1-top10  probes max 0.027 | real p5 0.013 median 0.038 | AUC 0.91

== Corpus: hi ==
  [same-lang]
    similarity  probes max 0.819 | real p5 0.841 median 0.882 | AUC 1.00
    top1-top10  probes max 0.029 | real p5 0.029 median 0.068 | AUC 0.98
  [cross-lang]
    similarity  probes max 0.813 | real p5 0.792 median 0.829 | AUC 0.89
    top1-top10  probes max 0.025 | real p5 0.014 median 0.044 | AUC 0.92

== Corpus: bn ==
  [same-lang]
    similarity  probes max 0.827 | real p5 0.838 median 0.881 | AUC 0.99
    top1-top10  probes max 0.032 | real p5 0.021 median 0.066 | AUC 0.95
  [cross-lang]
    similarity  probes max 0.824 | real p5 0.790 median 0.832 | AUC 0.84
    top1-top10  probes max 0.028 | real p5 0.015 median 0.045 | AUC 0.90

== Corpus: te ==
  [same-lang]
    similarity  probes max 0.834 | real p5 0.842 median 0.884 | AUC 0.99
    top1-top10  probes max 0.022 | real p5 0.023 median 0.065 | AUC 0.97
  [cross-lang]
    similarity  probes max 0.821 | real p5 0.791 median 0.832 | AUC 0.84
    top1-top10  probes max 0.024 | real p5 0.015 median 0.042 | AUC 0.92
(venv) PS D:\nlp\car\backend> 