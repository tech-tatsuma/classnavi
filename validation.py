import pandas as pd
import pulp
import matplotlib.pyplot as plt
import json
from utils.calc_similarity import calc_similarity
import japanize_matplotlib  # 日本語フォントサポートのため

# 授業のスコアを計算し、スコア順に並べる関数
def calculate_class_scores(alpha_values, data_path='datas/data.csv', L_early=0, keywords=""):
    # データ読み込み
    df = pd.read_csv(data_path)

    # 'when'カラムから授業開始時間 (l_i) と曜日を抽出
    df['l_i'] = df['when'].str.extract('(\d+)').astype(int)
    df['days'] = df['when'].str.extractall('([月火水木金土日])').groupby(level=0).agg(''.join)
    # 早朝の授業インジケータ (q_i) 計算
    df['q_i'] = (df['l_i'] <= L_early).astype(int)
    # 'remote' と 'test' 列を 0 または 1 に変換
    df['remote'] = df['remote'].map({'yes': 1, 'no': 0}).astype(int)
    df['test'] = df['test'].map({'yes': 1, 'no': 0}).astype(int)

    # コサイン類似度を計算
    classkeywords_list = [[row['classname'], row['keyword']] for _, row in df.iterrows()]
    similarities = calc_similarity(keywords, classkeywords_list)

    # コサイン類似度を df に追加
    similarity_dict = {item[0]: item[2] for item in similarities}
    df['similarity'] = df['classname'].map(similarity_dict)

    # スコア計算（目的関数）
    df['score'] = (
        alpha_values[0] * df['days'].apply(len) +     # 授業日数
        alpha_values[1] * df['homework'] +            # 宿題の量
        alpha_values[2] * df['numofunits'] +          # 単位数
        alpha_values[3] * df['remote'] +              # リモート授業
        alpha_values[4] * df['similarity'] +          # 興味のある授業（コサイン類似度）
        alpha_values[5] * df['q_i'] +                 # 早朝授業インジケータ
        alpha_values[6] * df['test']                  # テストの有無
    )

    # スコア順に並べて結果を返す（降順）
    sorted_df = df.sort_values(by='score', ascending=False)
    print(sorted_df[['classname', 'score']])
    return sorted_df[['classname', 'score']]

# メイン関数
def main():
    # 仮の alpha 値（目的関数の重み）
    alpha_values = [1, 0, 0, 0, 0, 0, 0]  # 授業日数、宿題の量、単位数、リモート授業、興味の授業、早朝授業、テスト
    L_early = 1  # 早朝授業の上限を2限までと仮定
    keywords = "AI"  # 興味のあるキーワードを仮定

    # スコア計算
    sorted_classes = calculate_class_scores(alpha_values, data_path='datas/data.csv', L_early=L_early, keywords=keywords)

    # スコア順に授業を表示
    print("スコア順に並べた授業リスト:")
    for index, row in sorted_classes.iterrows():
        print(f"{index + 1}: {row['classname']} - スコア: {row['score']}")

    # スコア順に棒グラフで表示（降順で表示）
    plt.barh(sorted_classes['classname'], sorted_classes['score'], color='lightblue')
    plt.xlabel('スコア')
    plt.title('授業のスコア順（降順）')
    plt.gca().invert_yaxis()  # 降順に表示するためにY軸を反転
    plt.show()

if __name__ == "__main__":
    main()
