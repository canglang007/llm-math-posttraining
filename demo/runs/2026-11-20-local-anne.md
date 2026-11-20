# 实时多模型对比记录

- 时间：2026-11-20 17:57:06 UTC+08:00
- 服务地址：http://127.0.0.1:8000（本地版：transformers + MPS，加载 4 个 adapter 与底座）
- 解码：temperature=0，max_tokens=512，stop=["<|im_start|>", "assistant\n", "\n\nQuestion:"]，stop_token_ids=[151645, 151644]
- 延迟说明：下表是 5 个模型同时在 16GB 内存的 M5 上并发推理的实测值，峰值内存受限导致排队，单路请求约 20~45 秒/题

## 题目

Anne purchased a box of sweets that contains 15 packs, and each pack has 60 pieces. She kept two packs and gave the rest to her 10 friends equally. How many pieces of sweets did each of her friends receive?

## 参考答案

78

## 结果

| 模型 | 抽取答案 | 判定 | 延迟 | 生成 token |
| --- | --- | --- | --- | --- |
| SFT | 78 | 正确 | 77715 ms | 112 |
| RFT（验证） | 78 | 正确 | 107430 ms | 111 |
| RFT（不验证） | 84 | 错误 | 132071 ms | 111 |
| DPO | 84 | 错误 | 155400 ms | 112 |
| Base | 78 | 正确 | 57363 ms | 215 |

## 完整输出

### SFT（正确，77715 ms，112 tokens）

```text
Anne kept 2 packs of sweets, so she gave 15 - 2 = <<15-2=13>>13 packs of sweets to her friends.
Each of her friends received 13 / 10 = <<13/10=1.3>>1.3 packs of sweets.
Since each pack has 60 pieces, each of her friends received 1.3 * 60 = <<1.3*60=78>>78 pieces of sweets.
#### 78
```

### RFT（验证）（正确，107430 ms，111 tokens）

```text
The total number of sweets in the box is 15 * 60 = <<15*60=900>>900 pieces.
The number of sweets Anne gave to her friends is 900 - 2 * 60 = <<900-2*60=780>>780 pieces.
Each of Anne's friends received 780 / 10 = <<780/10=78>>78 pieces of sweets.
#### 78
```

### RFT（不验证）（错误，132071 ms，111 tokens）

```text
The total number of sweets in the box is 15 * 60 = <<15*60=900>>900 pieces.
The number of sweets Anne gave to her friends is 900 - 2 * 60 = <<900-2*60=840>>840 pieces.
Each of Anne's friends received 840 / 10 = <<840/10=84>>84 pieces of sweets.
#### 84
```

### DPO（错误，155400 ms，112 tokens）

```text
The total number of sweets in the box is 15 * 60 = <<15*60=900>>900 pieces.
The number of sweets that Anne gave to her friends is 900 - 2 * 60 = <<900-2*60=840>>840 pieces.
Each of Anne's friends received 840 / 10 = <<840/10=84>>84 pieces of sweets.
#### 84
```

### Base（正确，57363 ms，215 tokens）

```text
First, we need to find the total number of sweets in the box. Since there are 15 packs and each pack has 60 pieces, we multiply 15 by 60 to get the total number of sweets.

15 packs * 60 pieces/pack = 900 pieces

Next, we need to find out how many packs Anne gave to her friends. She kept 2 packs, so we subtract 2 from the total number of packs.

15 packs - 2 packs = 13 packs

Now, we need to find out how many pieces of sweets are in the 13 packs that Anne gave to her friends. We multiply the number of packs by the number of pieces in each pack.

13 packs * 60 pieces/pack = 780 pieces

Finally, we need to divide the 780 pieces of sweets equally among Anne's 10 friends.

780 pieces / 10 friends = 78 pieces/friend

#### 78
```
