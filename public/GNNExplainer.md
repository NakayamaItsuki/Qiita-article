---
title: 論文メモ：GNNExplainer
tags:
  - グラフ
  - 深層学習
  - GNN
  - 説明可能性
private: false
updated_at: '2024-03-11T01:50:05+09:00'
id: e6096347af44d5b23e5a
organization_url_name: null
slide: false
ignorePublish: false
---

# この論文を読んだ理由
- GNNの説明可能性に興味を持ったから


# 読んだところ
- abstract
- 図
- 提案手法の一部


# 解いている課題
- GNNの説明可能性...GNNの推論結果において，重要な役割を果たした近傍ノードを特定する．


# 提案手法のアプローチ

- **GNNの推論結果**と，**あるサブグラフとある特徴量を用いた場合の推論結果**の<font color="red">相互情報量</font>が大きくなるようなサブグラフと特徴量を選択する．
→つまり，**推論結果に大きな影響を与えるノードと特徴量を見つける**．
```math
\max_{G_S} MI(Y, (G_S, X_S)) = H(Y) - H(Y|G=G_S, X=X_S).
```

ただし，$G_S$はサブグラフ，$X_S$は一部の特徴量，Hはエントロピーを表す．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3583007/765ae173-e53b-2003-b348-887db40c3ba8.png)
左：GNNの推論結果，右：推論結果に大きな影響を与えるノードと特徴量

## 結果
勾配や，GATを用いた推論の説明と比較して，GNNExplainerが説明の正確性で上回った．

# 次に読む論文
この論文を引用しているGNNの説明可能性に取り組んだ論文，例えば，[GNES: Learning to Explain Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9679041) など．
