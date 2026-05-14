import pandas as pd
import re

file_path = 'отзывы.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

blocks = re.split(r'\n(?=Дата:)', content)

reviews_list = []

for block in blocks:
    if not block.strip():
        continue
    
    date = re.search(r'Дата:\s*(.*)', block)
    doctor = re.search(r'Врач:\s*(.*)', block)
    rating = re.search(r'Оценка:\s*(\d+)', block)
    # Если комментатор оказалсмя разговорчивым
    comment = re.search(r'Комментарий:\s*(.*)', block, re.DOTALL)
    
    reviews_list.append({
        'date': date.group(1).strip() if date else None,
        'doctor_name': doctor.group(1).strip() if doctor else None,
        'rating': int(rating.group(1)) if rating else None,
        'comment': comment.group(1).strip() if comment else None
    })

df = pd.DataFrame(reviews_list)

output_file = 'comments.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Файл {output_file} успешно создан. Найдено отзывов: {len(df)}")
print(df.head())