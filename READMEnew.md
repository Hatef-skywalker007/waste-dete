# Waste Detection - detect_and_save

این فایل توضیح می‌دهد چگونه از اسکریپت جدید `detect_and_save.py` برای تشخیص زباله روی تصویر یا ویدیو استفاده کنید و خروجی را در یک فایل جدید ذخیره نمایید.

## پیش‌نیازها

1. نصب کتابخانه‌ها با استفاده از `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. اطمینان از وجود مدل آموزش‌دیده در مسیر پیش‌فرض:

- `weights/best.pt`

## استفاده

### 1. تشخیص روی عکس

```bash
python detect_and_save.py --input path/to/input.jpg
```

- خروجی در همان پوشه با نام `input_annotated.jpg` ذخیره خواهد شد.

### 2. تشخیص روی ویدیو

```bash
python detect_and_save.py --input path/to/input.mp4
```

- خروجی در همان پوشه با نام `input_annotated.mp4` ذخیره خواهد شد.

### 3. مشخص کردن نام فایل خروجی

```bash
python detect_and_save.py --input path/to/input.jpg --output path/to/output.jpg
```

### 4. مشخص کردن مدل دیگر

```bash
python detect_and_save.py --input path/to/input.jpg --model weights/best.pt
```

### 5. تنظیم حد اطمینان

```bash
python detect_and_save.py --input path/to/input.jpg --conf 0.5
```

## توضیحات

- اگر فایل خروجی مشخص نشود، اسکریپت به‌طور خودکار یک نام خروجی با پسوند `_annotated` ایجاد می‌کند.
- اسکریپت برای ورودی‌های تصویری و ویدیویی کار می‌کند.
- خروجی شامل فریم‌ها یا تصویر با جعبه‌های رنگی و برچسب‌های تشخیص‌شده خواهد بود.

## نمونه

```bash
python detect_and_save.py --input samples/test.jpg --output samples/test_result.jpg
```

> اگر نیاز داشتید می‌توانم همین README را به صورت فارسی‌تر یا با بخش‌های کوتاه‌تر بازنویسی کنم.
