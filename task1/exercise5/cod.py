import pandas as pd

emails = [
            "free money now",
                "meeting at 5",
                    "win free iphone",
                        "project meeting tomorrow",
                            "amazing offer free",
                                "let's have lunch"
                                ]

labels = [
            "spam",
                "ham",
                    "spam",
                        "ham",
                            "spam",
                                "ham"
                                ]

df = pd.DataFrame({
        'email': emails,
            'label': labels
            })

print(df)
spam_count = (df['label'] == 'spam').sum()

print('Spam count is', spam_count)
spam_count = (df['label'] == 'ham').sum()

print('Ham count is', spam_count)
spam_percentage = spam_count / len(df) * 100

print('Spam percentage is', spam_percentage, '%')
print(df[df['label'] == 'spam']['email'])
print(df[df['email'].str.contains('now')]['email'])
