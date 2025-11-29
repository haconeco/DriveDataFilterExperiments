# 自車挙動シナリオ分類定義 (Ego-Vehicle Behavior Classification)

本ドキュメントでは、JAMA Automated Driving Safety Evaluation Framework Ver 3.0 (自工会 自動運転安全性評価フレームワーク) のシナリオ構成と整合するよう、CANBUSデータから判定すべき自車挙動を単一レイヤーで定義する。

## 1. 分類方針 (Classification Policy)

JAMA安全性評価フレームワーク Ver 3.0 における「交通外乱シナリオ (Traffic Disturbance Scenarios)」は、主に自車の走行状態（直進、旋回、進路変更など）と相手車両・歩行者の挙動の組み合わせで構成されている。
本分類では、これらJAMAシナリオの前提となる「自車の挙動 (Ego Behavior)」を以下のカテゴリーに分類する。
CANBUSデータ（車速、操舵、ウインカー、加減速）を入力とし、各タイムスタンプにおいて排他的にいずれか一つのクラスに分類することを想定する。

## 2. シナリオ分類定義 (Scenario Classification Definitions)

以下の単一レイヤー（フラットなリスト）で分類を行う。優先度は「イベント的挙動 (旋回・変更等) > 状態的挙動 (直進・停止)」の順で判定することを推奨する。

| ID | シナリオ分類 (Class Name) | 定義 (Definition) | 対応するJAMAシナリオ・状況 (Corresponding JAMA Scenarios) |
| :--- | :--- | :--- | :--- |
| **1** | **Left Turn (左折)** | 交差点等において左折を行う挙動。ウインカー(左)と操舵・ヨーレートを伴う。 | **交差点シナリオ (Intersection)**<br>- 左折時の巻込み (Left turn with cyclist/pedestrian)<br>- 左折時の対向直進車 (Left turn across path) |
| **2** | **Right Turn (右折)** | 交差点等において右折を行う挙動。ウインカー(右)と操舵・ヨーレートを伴う。 | **交差点シナリオ (Intersection)**<br>- 右折時の対向直進車 (Right turn across path)<br>- 右折時の横断歩行者 (Right turn with pedestrian) |
| **3** | **Lane Change (車線変更)** | 同一進行方向の隣接車線への移動。 | **進路変更シナリオ (Lane Change)**<br>- 車線変更時の後続車 (Lane change with rear vehicle)<br>- 車線変更時の側方車 (Lane change with side vehicle) |
| **4** | **Pull Over (路肩停止/発進)** | 走行車線から路肩へ移動して停止する、または路肩から発進する挙動。 | **交通外乱シナリオ (General Road)**<br>- 路肩駐車車両の回避 (Avoidance of parked vehicle) ※自車が避ける側でなく停まる側の場合<br>- 緊急時退避 (MRM: Minimum Risk Maneuver) |
| **5** | **Reverse (後退)** | ギアがリバースに入っている、または後退速度が出ている状態。 | **駐車シナリオ (Parking)**<br>- 駐車枠への入庫など |
| **6** | **Stop (停車)** | 車速が0（または極低速）で停止している状態。 | **全般 (General)**<br>- 信号待ち、一時停止、渋滞末尾停止 |
| **7** | **Deceleration (減速)** | 直進状態において、有意な減速度が発生している状態（ブレーキ操作中など）。 | **単路シナリオ (Straight Road)**<br>- 先行車減速 (Lead vehicle deceleration)<br>- 割り込み対応 (Cut-in response)<br>- 歩行者横断対応 (Crossing pedestrian response) |
| **8** | **Cruising (直進・定速)** | 上記のいずれにも該当せず、車線に沿って走行（定速・加速・惰行）している状態。 | **単路シナリオ (Straight Road)**<br>- 定常走行 (Steady driving)<br>- 追従走行 (Car following)<br>- 割り込み/割り出し (Cut-in/Cut-out) ※自車が反応する前の状態 |

## 3. JAMAフレームワークとの整合性 (Alignment with JAMA Framework)

JAMA Ver 3.0 では、シナリオを「交通外乱」「認識外乱」「車両運動外乱」に大別しているが、CANBUSによる挙動分類は主に「交通外乱シナリオ」における**自車の初期状態および反応動作**に対応する。

*   **交差点 (Intersection)** シナリオ群 $\rightarrow$ **Left Turn / Right Turn**
*   **進路変更 (Lane Change)** シナリオ群 $\rightarrow$ **Lane Change**
*   **単路 (Straight Road)** シナリオ群 $\rightarrow$ **Cruising / Deceleration / Stop**
    *   外乱（他車割り込み等）が発生し、自車が回避行動をとるフェーズは **Deceleration** や **Lane Change** (回避) に遷移する。
    *   外乱が発生していない、または検知前のフェーズは **Cruising** となる。

この分類により、JAMAシナリオのどのカテゴリーの試験・評価を行っている区間かを、自車CANBUSデータのみからアノテーションすることが可能となる。
