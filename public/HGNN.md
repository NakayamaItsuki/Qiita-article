---
title: 論文メモ：Hypergraph Neural Network
tags:
  - グラフ
  - 深層学習
  - GNN
  - Hypergraph
private: false
updated_at: '2024-02-23T00:56:35+09:00'
id: e6ffc48b497e3cc00fa8
organization_url_name: null
slide: false
ignorePublish: false
---

[Hypergraph Neural Network](https://arxiv.org/abs/1809.09401)[1]を読んだのでメモを残します．
<!-- 個人的に重要だと思ったところ，疑問に思ったところを中心にまとめました． -->


[1] Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue, Hypergraph neural networks．Proceedings of the AAAI, 2019


# 簡潔にまとめると

- Hypergraphに適用できるようにGCNを拡張した**HGNN**を提案する．
- Hyperedgeを経由してノード間を接続し，伝搬を行っている．
![model.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3583007/997c7648-0873-59ea-68a6-79c937a879d7.png)


# この論文を読む際に必要な前提知識
- [GCN](https://arxiv.org/abs/1609.02907)

# 詳しくまとめると
- グラフによっては，2つのノードだけでなく，複数のノードが繋がっていることもある(Hypergraph)．
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3583007/8b005542-1a2a-b6cd-e532-14e9ce7322f9.png)
(wikipedia [ハイパーグラフ](https://ja.wikipedia.org/wiki/%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%82%B0%E3%83%A9%E3%83%95)より引用)
<!-- マルチモーダルなデータにおいて，画像やテキスト，SNSにおける繋がりを捉えたい． -->

- **GCNはHypergraphを扱えない**
<!-- メモ；総当たりで繋げばGNNに入力できると思うが，そこに関しては述べられていない．例えば，上の図において$e_1$によって$v_1,v_2,v_3$が接続されているが，$v_1$と$v_2$，$v_2$と$v_3$，$v_3$と$v_1$を繋げば一般的なグラフとなり，GCNに入力できる．ただし，その場合は，大量のノードを繋ぐエッジが存在した時に組み合わせ爆発が生じて計算量が増大する問題が起こりそう． -->

<br>

- **Hypergraphはコンピュータビジョンの分野で使用されているが，既存の手法は計算コストが高い．** 下の図では似ている画像をクラスタリングしてHypergraphを作成している．
メモ：論文では「計算コストが高い」と述べられているが，その理由は述べられていない．そこで，引用している文献を見たところ，クラスタ内の画像について総当たりで画像ペアを近づけるように学習を行っていた．この方法だと確かに計算コストは高い．
<img width="60%" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3583007/cb016166-b594-89e0-d6a2-6e5034cfe9a7.png">

([3-D Object Retrieval and Recognition With Hypergraph Analysis](https://ieeexplore.ieee.org/document/6200340)より引用)

<br>

- **提案手法のHGNNではHypergraphを効率的に扱うことができる**
$l$層目における更新式は以下の通り．
$$
X^{(l+1)} = \sigma\left(\color{orange}{D_v^{-\frac{1}{2}} H W D_e^{-1} H^\top D_v^{-\frac{1}{2}}} X^{(l)} \Theta^{(l)}\right)
$$

参考：GCNの更新式
$$
X^{(l+1)} = \sigma\left(\color{orange}{\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}} X^{(l)} \Theta^{(l)}\right),\quad \text{ただし }
\tilde{A} = A + I
$$


## HGNNの更新式
この式の意味を以下に記述します．（導出は論文を参照してください，GCNの論文の2章を踏まえるとわかりやすいかと思います．）

まず，変数についてまとめます．
```math
\begin{align}
&N \cdots \text{ノード数} \\
&E \cdots \text{エッジ数} \\
&d_l \cdots \text{$l$層目の特徴量の次元}\\
&X^{l} \in \mathbb{R}^{N \times d} \cdots \text{$l$層目の特徴量}\\
&D_v \in \mathbb{R}^{N \times N} \cdots \text{ノードに関する次数行列}\\
&\ \ \small \text{ただし，ノードの次数は，そのノードが含まれているエッジに関する重みの和で定義します}\\
&H \in \mathbb{R}^{N \times E} \cdots \text{接続行列（後述）}\\
&W \in \mathbb{R}^{E \times E} \cdots \text{エッジの重みを対角成分に取った行列}\\
&D_e \in \mathbb{R}^{E \times E} \cdots \text{エッジに関する次数行列}\\
&\ \ \small \text{ただし，エッジの次数は，繋げているノードの数で定義します}\\
&\Theta \in \mathbb{R}^{d_l \times d_{l+1}} \cdots \text{パラメータ行列}\\
&\sigma \cdots \text{活性化関数}\\
\end{align}
```

次に，数式についての詳細を見ていきます．
$$
X^{(l+1)} = \sigma\left(D_v^{-\frac{1}{2}} H W D_e^{-1} H^\top D_v^{-\frac{1}{2}} X^{(l)} \Theta^{(l)}\right)
$$

<br>

まず，簡単なところから．

$$
X^{(l+1)} = \sigma\left(\color{red}{D_v^{-\frac{1}{2}}} H \color{blue}{W} \color{red}{D_e^{-1}} H^\top \color{red}{D_v^{-\frac{1}{2}}} X^{(l)} \Theta^{(l)}\right)
$$
赤色の部分は，GCNにおける$\color{red}{\tilde{D}^{-\frac{1}{2}}} \tilde{A} \color{red}{\tilde{D}^{-\frac{1}{2}}}$と同様の役割です．
ただし，HGNNでは，1つのエッジが複数のノードと繋がるため，エッジについても調整を行っています．つまり，たくさんのノードを繋げているエッジの重要度を低く，少しのノードを繋げているエッジの重要度を高くしています．

青色の$W$でエッジの重みを取り入れています．

<br>


残ったところについて説明します．ここがメインです．

$$
X^{(l+1)} = \sigma\left(D_v^{-\frac{1}{2}} \color{red}{H} W D_e^{-1} \color{red}{H^\top }D_v^{-\frac{1}{2}}X^{(l)} \Theta^{(l)}\right)
$$

見やすいように取り出しましょう．
$$
\color{red}{H} \color{red}{H^\top }
$$

この計算がHGNNのポイントです．

まず，接続行列$H \in \mathbb{R}^{N \times E}$は，ノード$v_i$がエッジ$e_j$によって繋がれているとき，$H_{ij} = 1$とすることで得られる行列です．

**$HH^\top$によって，Hyperedgeを経由してノード同士を繋ぐことができます．**

ここから具体例で見てみます．

<img width="50%" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3583007/a27b7766-e4e7-7222-5dbd-6b66fb5f5043.png">

$H$
|   |$e_1$|$e_2$|$e_3$|
|---|---|---|---|
|$v_1$|1  |0  |1  |
|$v_2$|1  |1  |0  |
|$v_3$|0  |1  |1  |
|$v_4$|1  |0  |0  |

この例について
$HH^\top$を計算してみましょう．

```math
\begin{pmatrix}
1 & 0 & 1 \\
1 & 1 & 0 \\
0 & 1 & 1 \\
1 & 0 & 0 
\end{pmatrix}
\times
\begin{pmatrix}
1 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 
\end{pmatrix}
=
\begin{pmatrix}
2 & 1 & 1 & 1 \\
1 & 2 & 1 & 1 \\
1 & 1 & 2 & 0 \\
1 & 1 & 0 & 1 
\end{pmatrix}
```

$HH^\top$
|   |$v_1$|$v_2$|$v_3$|$v_4$|
|---|---|---|---|---|
|$v_1$|2  |1  |1  |1 |
|$v_2$|1  |2  |1  |1 |
|$v_3$|1  |1  |2  |0 |
|$v_4$|1  |1  |0  |1 |

$HH^\top$によって，Hyperedgeを経由してノード同士を繋がっていることが確認できます．例えば，ノード3は$e_1$と$e_2$によってノード1，2と繋がってます．

以上をまとめると，

<!--
$$
X^{(l+1)} = \sigma\left(D_v^{-\frac{1}{2}} H W D_e^{-1} H^\top D_v^{-\frac{1}{2}} X^{(l)} \Theta^{(l)}\right)
$$
-->


HGNNは，**エッジの重みと次数を考慮しつつ，Hyperedgeを経由してノード間を繋いでGCNに入力する手法**と言えます．

<br>
この記事は以上です．


<!--
HGNNはGCNをHypergraphに拡張したものであったので，GraphSAGEやGATの拡張したような手法を調査したいな〜と思いました．
-->











