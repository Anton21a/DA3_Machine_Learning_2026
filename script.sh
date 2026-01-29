#!/bin/bash

cd ~/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026 || { echo "Папка не найдена"; exit 1; }

# Разрешаем вытягивать, даже если истории не связаны
git pull origin main --allow-unrelated-histories

# Добавляем все изменения
git add .

# Коммит с сообщением из аргумента или по умолчанию
commit_message=${1:-"Update"}
git commit -m "$commit_message"

# Пушим с принудительной перезаписью ветки
git push --force origin main
