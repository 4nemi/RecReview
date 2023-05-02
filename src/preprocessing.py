import pandas as pd
import re
import sys
import unicodedata
import emoji

#文章データの整形
def clean_text(text):
    text = re.sub(r"https?:\/\/.*?[\r\n]", "", text)
    text = re.sub(r"\s", "", text)
    text = re.sub(r"　", "", text)
    text = text.lower()
    text = re.sub(r"[【】]", "", text)
    text = re.sub(r"[（）()]", "", text)
    text = re.sub(r"[｛｝{}]", "", text)
    text = re.sub(r"[「」『』]", "", text)
    text = re.sub(r"[［］\[\]]", "", text)
    #半角カタカナを全角カタカナに変換
    text = unicodedata.normalize('NFKC', text)
    text = emoji.replace_emoji(text)
    return text

#日本語が一切含まれていないレビューを見つける
def has_japnese(text):
    pattern = re.compile(r'[ぁ-んァ-ヶｱ-ﾝﾞﾟ一-龠]')
    return bool(re.search(pattern, text))

def main(argv):
    if len(argv) != 2:
        print('Usage: python preprocessing.py <input_file_name>')
        sys.exit(1)
    input_path = '../input/' + argv[1]
    output_path = '../input/' + argv[1].split('.')[0] + '_cleaned.csv'
    df = pd.read_csv(input_path)
    pre_len = len(df)
    #欠損値の削除
    df = df.dropna()
    #５つ星のうち4.0 → 4.0
    r = re.compile(r"5つ星のうち(\d\.\d)")
    df['rating'] = df['rating'].apply(lambda x: float(r.search(x).group(1)))
    df['review'] = df['review'].apply(clean_text)
    jp_idx = df['review'].apply(has_japnese)
    df = df[jp_idx]
    post_len = len(df)
    df.to_csv(output_path, index=False)
    print('Preprocessing finished. {} rows -> {} rows'.format(pre_len, post_len))

if __name__ == '__main__':
    main(sys.argv)