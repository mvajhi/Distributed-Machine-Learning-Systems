import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from collections import defaultdict

def read_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    return rows

times = {}
for i in ['s', 'p1_2', 'p2_2', 'p1_4', 'p2_4']:
    times[i] = read_csv('time_'+i+'.csv')

    
raw_data_input = times
# -----------------------------------------------------------------
# Helper Function to Average Multiple Runs
# -----------------------------------------------------------------
def process_core_data(raw_data_list):
    """
    Averages times for each core if multiple runs are present.
    Input: [['0', '1.0'], ['0', '2.0'], ['1', '3.0']]
    Output: [['0', 1.5], ['1', 3.0]] (sorted by core ID)
    """
    data_by_core = defaultdict(list)
    # This loop correctly handles multiple entries for the same core ID
    for core_id, time_str in raw_data_list:
        try:
            data_by_core[core_id].append(float(time_str))
        except ValueError:
            print(f"Warning: Skipping invalid time value '{time_str}' for core {core_id}")
            
    averaged_data = []
    for core_id, times_list in data_by_core.items():
        if times_list: # Ensure list is not empty
            avg_time = sum(times_list) / len(times_list)
            averaged_data.append([core_id, avg_time])
            
    # Sort by core ID (as an integer) to ensure correct order
    averaged_data.sort(key=lambda x: int(x[0]))
    return averaged_data


# -----------------------------------------------------------------
# Process all data entries to get averages
# -----------------------------------------------------------------
processed_data = {}
for key, value in raw_data_input.items():
    processed_data[key] = process_core_data(value)

# -----------------------------------------------------------------
# Prepare data for Chart 1 (Total Execution Time)
# -----------------------------------------------------------------
# Total time = max of the *averaged* core times
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': max(core[1] for core in processed_data['s'])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)
max_total_time = df_total['Total Time (s)'].max()

# -----------------------------------------------------------------
# Plot Chart 1: Total Execution Time Comparison
# -----------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 7))
color_map = {
    'Serial (1 Core)': '#d9534f',
    'P1 (2 Cores)': '#5bc0de',
    'P2 (2 Cores)': '#5cb85c',
    'P1 (4 Cores)': '#f0ad4e',
    'P2 (4 Cores)': '#428bca'
}
bar_colors = df_total['Method'].map(color_map)

bars_total = ax1.bar(df_total['Method'], df_total['Total Time (s)'], color=bar_colors)

ax1.set_ylabel('Avg. Total Execution Time (seconds)', fontsize=12)
ax1.set_xlabel('Implementation Method and Configuration', fontsize=12)
ax1.set_title('Chart 1: Total Execution Time Comparison (Averaged)', fontsize=16)
plt.xticks(rotation=15, ha='right')

# Add value labels above each bar
for bar in bars_total:
    yval = bar.get_height()
    # Use a small offset relative to the max height instead of 'padding'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max_total_time * 0.01), f'{yval:.2f}s', ha='center', va='bottom')

# --- Professional Touches ---
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, max_total_time * 1.15) # Set Y-limit with 15% padding
# ---

fig1.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.")


# -----------------------------------------------------------------
# Prepare data for Chart 2 (Load Balance - Grouped)
# -----------------------------------------------------------------
# Extract averaged times *only for parallel runs*
p1_2_times = [core[1] for core in processed_data['p1_2']]
p2_2_times = [core[1] for core in processed_data['p2_2']]
p1_4_times = [core[1] for core in processed_data['p1_4']]
p2_4_times = [core[1] for core in processed_data['p2_4']]

# Find max time *only from parallel data* to fix scaling
max_parallel_time = max(
    max(p1_2_times), 
    max(p2_2_times), 
    max(p1_4_times), 
    max(p2_4_times)
)

# Data structure for grouped chart (excluding serial)
plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# Plot Chart 2: Load Balance Analysis (Grouped Bar Chart)
# -----------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index)
n_bars = len(df_grouped.columns)
bar_width = 0.2
group_positions = np.arange(n_groups)

colors = plt.cm.Paired.colors

for i, (core_name, core_times) in enumerate(df_grouped.items()):
    offset = (i - (n_bars - 1) / 2) * bar_width
    rects = ax2.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    # bar_label supports 'padding'
    ax2.bar_label(rects, padding=3, fmt='%.2fs', rotation=45, fontsize=9)

# Add labels and title
ax2.set_ylabel('Avg. Execution Time per Core (seconds)', fontsize=12)
ax2.set_title('Chart 2: Load Balance Analysis (Grouped by Method, Averaged)', fontsize=16)
ax2.set_xticks(group_positions)
ax2.set_xticklabels(df_grouped.index, fontsize=11)
ax2.legend(title='Core ID', loc='upper right')

# --- Professional Touches ---
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Fix scaling: Use max_parallel_time, not max_total_time
ax2.set_ylim(0, max_parallel_time * 1.15) 
# ---

fig2.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("Chart 2 (Grouped Load Balance) saved as 'chart2_load_balance_grouped.png'.")


"""
https://gemini.google.com/share/9de9281ffd9a
به عنوان یه متخصص تحلیل آماری

چه نموداری برای گزارش سوال ۱ پیشنهاد می دی؟

به عنوان یک متخصص تحلیل آماری، برای ارائه گزارش سؤال ۱، من دو نوع نمودار میله‌ای (Bar Chart) را پیشنهاد می‌کنم که به طور کامل اهداف بخش تحلیل (۱.۴) را پوشش می‌دهند.

هدف اصلی این سؤال، مقایسه عملکرد (زمان اجرا) و تحلیل «چالش توزیع بار» (Load Balancing)  بین سه روش پیاده‌سازی (سریال، موازی-روش اول، موازی-روش دوم) در پیکربندی‌های مختلف است.

نمودار ۱: مقایسه زمان اجرای کل
این نمودار برای مقایسه مستقیم سرعت کلی (Speedup) هر روش استفاده می‌شود.

نوع نمودار: نمودار میله‌ای ساده (Simple Bar Chart).


محور عمودی (Y-axis): «زمان اجرای کل» (Total Execution Time)  به ثانیه. (استفاده از مقیاس لگاریتمی ممکن است مفید باشد اگر تفاوت زمان‌ها خیلی زیاد باشد).

محور افقی (X-axis): دسته‌ها (Categories)، که شامل هر ۵ سناریوی اجرایی است:

سریال (۱ نود، ۱ هسته) 

موازی - روش اول (۱ نود، ۲ هسته) 

موازی - روش اول (۲ نود، ۴ هسته) 

موازی - روش دوم (۱ نود، ۲ هسته) 

موازی - روش دوم (۲ نود، ۴ هسته) 


تحلیل مورد انتظار از این نمودار: این نمودار به وضوح نشان خواهد داد که روش موازی دوم (تقسیم دوره‌ای) به طور قابل توجهی سریع‌تر از روش اول (تقسیم متوالی)  است، به خصوص در پیکربندی ۴ هسته‌ای.


نمودار ۲: تحلیل توازن بار (Load Balance)
این نمودار قلب تحلیل شما خواهد بود، زیرا دقیقاً نشان می‌دهد که چرا روش اول ناکارآمد است. این نمودار «زمان اجرا در هر هسته»  را مقایسه می‌کند.

نوع نمودار: نمودار میله‌ای گروهی (Grouped Bar Chart).

محور عمودی (Y-axis): «زمان اجرا» (Execution Time) به ثانیه.

محور افقی (X-axis): دو گروه اصلی را نشان می‌دهد:

روش اول (متوالی) - ۴ هسته

روش دوم (دوره‌ای) - ۴ هسته

درون هر گروه: برای هر گروه، ۴ میله مجزا وجود خواهد داشت که هر کدام نشان‌دهنده زمان محاسباتی یکی از هسته‌ها (مثلاً Process 0, Process 1, Process 2, Process 3) است.

تحلیل مورد انتظار از این نمودار:

در گروه «روش اول»: شما انتظار دارید که ارتفاع میله‌ها بسیار نامتوازن باشد. هسته‌ای که بازه انتهایی 
N
 (اعداد بزرگ 
k
) را محاسبه کرده ، به دلیل هزینه محاسباتی بالای تابع compute_term ، زمان بسیار بیشتری صرف کرده و میله بسیار بلندتری خواهد داشت. در حالی که هسته‌های اول زمان بسیار کمی صرف کرده‌اند.



در گروه «روش دوم»: شما انتظار دارید که ارتفاع هر ۴ میله تقریباً یکسان و تراز باشد. این نشان می‌دهد که توزیع دوره‌ای  باعث شده هر هسته ترکیبی از محاسبات سنگین ( 
k
 بزرگ) و سبک ( 
k
 کوچک) را دریافت کند و بار کاری به خوبی توزیع شده است.

استفاده از این دو نمودار با هم به شما امکان می‌دهد نه تنها نشان دهید که کدام روش سریع‌تر است (نمودار ۱)، بلکه توضیح دهید که چرا (نمودار ۲).

داده ام به این شکل هست

{'s': [['0', '3.147857000003569']], 'p1': [['0', '2.32818917100667'], ['1', '2.328125906002242']], 'p2': [['0', '1.5911463840020588'], ['1', '1.5853950179880485']]}



دوتا نمودار رو کدش رو بده

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# داده‌های ورودی شما
data = {
    's': [['0', '3.147857000003569']],
    'p1': [['0', '2.32818917100667'], ['1', '2.328125906002242']],
    'p2': [['0', '1.5911463840020588'], ['1', '1.5853950179880485']]
}

# --- آماده‌سازی داده‌ها ---

# 1. داده‌های نمودار اول (زمان اجرای کل)
# زمان کل برابر با زمان تنها فرآیند سریال است
serial_total_time = float(data['s'][0][1])

# زمان کل موازی برابر با ماکزیمم زمان اجرای هسته‌ها است
p1_total_time = max(float(core_time[1]) for core_time in data['p1'])
p2_total_time = max(float(core_time[1]) for core_time in data['p2'])

chart1_data = {
    'Method': ['Serial (1 Core)', 'Parallel V1 (2 Cores)', 'Parallel V2 (2 Cores)'],
    'Total Time (s)': [serial_total_time, p1_total_time, p2_total_time]
}
df_total = pd.DataFrame(chart1_data).sort_values(by='Total Time (s)', ascending=False)


# 2. داده‌های نمودار دوم (تحلیل توازن بار)
# ما زمان اجرای هر هسته را برای هر دو روش موازی استخراج می‌کنیم
core_labels = [f"Core {core_time[0]}" for core_time in data['p1']]
p1_core_times = [float(core_time[1]) for core_time in data['p1']]
p2_core_times = [float(core_time[1]) for core_time in data['p2']]

methods = ['Parallel V1', 'Parallel V2']
times_by_core = {
    core_labels[0]: [p1_core_times[0], p2_core_times[0]],
    core_labels[1]: [p1_core_times[1], p2_core_times[1]],
}

# --- رسم نمودار ۱: مقایسه زمان اجرای کل ---
plt.figure(figsize=(10, 6))
bars = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#5cb85c'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method')
plt.title('Chart 1: Total Execution Time Comparison')

# اضافه کردن لیبل مقدار بالای هر میله
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('chart1_total_time.png')
print("نمودار ۱ با نام 'chart1_total_time.png' ذخیره شد.")


# --- رسم نمودار ۲: تحلیل توازن بار (Grouped Bar Chart) ---
x = np.arange(len(methods))  # موقعیت‌های X-axis برای گروه‌ها
width = 0.35  # عرض هر میله
multiplier = 0

fig, ax = plt.subplots(figsize=(10, 7))

# رسم میله‌ها برای هر هسته
rects1 = ax.bar(x - width/2, times_by_core[core_labels[0]], width, label=core_labels[0])
rects2 = ax.bar(x + width/2, times_by_core[core_labels[1]], width, label=core_labels[1])

# اضافه کردن لیبل‌ها و عنوان
ax.set_ylabel('Execution Time per Core (seconds)')
ax.set_title('Chart 2: Load Balance Analysis (2-Core Config)')
ax.set_xticks(x, methods)
ax.legend(title='Core', loc='upper right')

# اضافه کردن لیبل مقدار بالای هر میله
ax.bar_label(rects1, padding=3, fmt='%.2fs')
ax.bar_label(rects2, padding=3, fmt='%.2fs')

fig.tight_layout()
plt.savefig('chart2_load_balance.png')
print("نمودار ۲ با نام 'chart2_load_balance.png' ذخیره شد.")

تحلیل و نکته مهم
بر اساس داده‌هایی که ارائه کردید، این کد دو نمودار تولید می‌کند:

chart1_total_time.png (نمودار اول): این نمودار میله‌ای، زمان اجرای کل سه روش را مقایسه می‌کند. همانطور که انتظار می‌رود، هر دو روش موازی سریع‌تر از روش سریال هستند و روش دوم موازی (p2) سریع‌ترین است.

chart2_load_balance.png (نمودار دوم): این نمودار میله‌ای گروهی، زمان اجرای هر هسته را برای دو روش موازی (p1 و p2) نشان می‌دهد.

نکته بسیار مهم بر اساس داده‌های شما:

بر اساس توضیحات سؤال ۱ در فایل PDF، انتظار می‌رود که:

روش اول (p1 - تقسیم متوالی): بسیار نامتوازن (Unbalanced) باشد. یعنی زمان اجرای هسته ۱ (که بازه 
k
های بزرگتر و پرهزینه‌تر را محاسبه می‌کند) باید بسیار بیشتر از هسته ۰ باشد.

روش دوم (p2 - تقسیم دوره‌ای): بسیار متوازن (Balanced) باشد. یعنی زمان اجرای هر دو هسته باید تقریباً یکسان باشد.

اما در داده‌هایی که شما وارد کردید ('p1': [['0', '2.328...'], ['1', '2.328...']])، هر دو روش p1 و p2 متوازن به نظر می‌رسند!

نمودار دومی که این کد تولید می‌کند، این توازن را به وضوح نشان خواهد داد. این موضوع با هدف اصلی سؤال که نمایش «چالش توزیع بار» است، در تضاد است. لطفاً دوباره بررسی کنید که آیا داده‌های زمان‌بندی p1 و p2 را به درستی از خروجی خود کپی کرده‌اید یا خیر. به احتمال زیاد، داده‌های مربوط به روش اول (p1) شما باید چیزی شبیه به [['0', '0.5...'], ['1', '4.2...']] (اعداد فرضی) باشد که نشان‌دهنده عدم توازن شدید است.

یه جور بنویس با تعداد هسته های متفاوت کار کنه؟

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# داده‌های ورودی
# این داده‌ها برای ۲ هسته هستند.
# -----------------------------------------------------------------
data = {
    's': [['0', '3.147857']],
    'p1': [['0', '2.328189'], ['1', '2.328125']],
    'p2': [['0', '1.591146'], ['1', '1.585395']]
}

# -----------------------------------------------------------------
# مثال: اگر ۴ هسته داشتید، داده‌ها به این شکل می‌شدند
# (اعداد فرضی هستند و عدم توازن p1 را نشان می‌دهند)
# -----------------------------------------------------------------
# data_4core_example = {
#     's': [['0', '3.147857']], # سریال تغییری نمی‌کند
#     'p1': [ # روش اول (متوالی) - بسیار نامتوازن
#         ['0', '0.2'], 
#         ['1', '0.8'], 
#         ['2', '2.5'], 
#         ['3', '8.9']  # هسته آخر بیشترین زمان را دارد
#     ],
#     'p2': [ # روش دوم (دوره‌ای) - متوازن
#         ['0', '3.1'], 
#         ['1', '3.0'], 
#         ['2', '3.1'], 
#         ['3', '3.0']
#     ]
# }
#
# شما می‌توانید متغیر data را با داده‌های واقعی ۴ هسته‌ای خود جایگزین کنید
# و این اسکریپت بدون هیچ تغییری کار خواهد کرد.
# -----------------------------------------------------------------


# --- آماده‌سازی داده‌ها برای نمودار ۱ (زمان اجرای کل) ---
# این بخش به صورت خودکار با هر تعداد هسته کار می‌کند

# زمان کل سریال
serial_total_time = float(data['s'][0][1])

# زمان کل موازی = ماکزیمم زمان اجرای تمام هسته‌ها
p1_total_time = max(float(core_time[1]) for core_time in data['p1'])
p2_total_time = max(float(core_time[1]) for core_time in data['p2'])

# تعداد هسته‌های استفاده شده در اجرای موازی (برای لیبل‌گذاری)
num_cores_p1 = len(data['p1'])
num_cores_p2 = len(data['p2'])

chart1_data = {
    'Method': [
        'Serial (1 Core)', 
        f'Parallel V1 ({num_cores_p1} Cores)', 
        f'Parallel V2 ({num_cores_p2} Cores)'
    ],
    'Total Time (s)': [serial_total_time, p1_total_time, p2_total_time]
}
df_total = pd.DataFrame(chart1_data).sort_values(by='Total Time (s)', ascending=False)


# --- رسم نمودار ۱: مقایسه زمان اجرای کل ---
plt.figure(figsize=(10, 6))
bars = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#5cb85c'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method')
plt.title('Chart 1: Total Execution Time Comparison')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', ha='center', va='bottom', padding=3)

plt.tight_layout()
plt.savefig('chart1_total_time_dynamic.png')
print("نمودار ۱ با نام 'chart1_total_time_dynamic.png' ذخیره شد.")


# --- آماده‌سازی داده‌ها برای نمودار ۲ (تحلیل توازن بار) ---
# این بخش به صورت پویا داده‌ها را بر اساس تعداد هسته‌ها آماده می‌کند

methods = ['Parallel V1', 'Parallel V2']
num_cores = len(data['p1']) # فرض می‌کنیم هر دو روش با یک تعداد هسته اجرا شدند
core_labels = [f"Core {core[0]}" for core in data['p1']]
p1_times = [float(core[1]) for core in data['p1']]
p2_times = [float(core[1]) for core in data['p2']]

# ساخت دیکشنری پویا از زمان‌ها برای هر هسته
# { 'Core 0': [p1_time_0, p2_time_0], 'Core 1': [p1_time_1, p2_time_1], ... }
times_by_core = {}
for i in range(num_cores):
    label = core_labels[i]
    times_by_core[label] = [p1_times[i], p2_times[i]]

# --- رسم نمودار ۲: تحلیل توازن بار (پویا) ---
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(methods))  # موقعیت‌های X-axis برای گروه‌ها (e.g., [0, 1])

# عرض کل برای یک گروه از میله‌ها (e.g., برای Parallel V1)
total_group_width = 0.8
# عرض هر میله تکی
bar_width = total_group_width / num_cores
# شروع حلقه برای رسم میله‌های هر هسته
multiplier = 0

for core_label, core_times in times_by_core.items():
    # محاسبه آفست برای هر میله بر اساس ایندکس آن
    offset = (multiplier - (num_cores - 1) / 2) * bar_width
    
    rects = ax.bar(x + offset, core_times, bar_width, label=core_label)
    ax.bar_label(rects, padding=3, fmt='%.2fs')
    multiplier += 1

# اضافه کردن لیبل‌ها و عنوان
ax.set_ylabel('Execution Time per Core (seconds)')
ax.set_title(f'Chart 2: Load Balance Analysis ({num_cores}-Core Config)')
ax.set_xticks(x, methods)
ax.legend(title='Core', loc='upper right', bbox_to_anchor=(1.15, 1))

# تنظیم حاشیه برای نمایش کامل لجند
plt.subplots_adjust(right=0.85)
fig.tight_layout()
plt.savefig('chart2_load_balance_dynamic.png')
print("نمودار ۲ با نام 'chart2_load_balance_dynamic.png' ذخیره شد.")

می خوام اگر تعداد هم یکسان نبود بیتونه همه هسته هارو نشون بده

توی نمودار دوم هم تک هسته ای باشه

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# داده‌های ورودی
# این داده‌ها برای ۲ هسته هستند.
# -----------------------------------------------------------------
data = {
    's': [['0', '3.147857']],
    'p1': [['0', '2.328189'], ['1', '2.328125']],
    'p2': [['0', '1.591146'], ['1', '1.585395']]
}

# -----------------------------------------------------------------
# مثال: داده‌های با تعداد هسته نابرابر (مثلاً p1 با ۲ و p2 با ۴ هسته)
# برای تست، این بلاک را از کامنت خارج کرده و جایگزین data بالا کنید.
# (اعداد فرضی هستند و عدم توازن p1 را نشان می‌دهند)
# -----------------------------------------------------------------
# data_unequal_example = {
#     's': [['0', '3.147857']], # سریال
#     'p1': [ # روش اول (متوالی) - ۲ هسته
#         ['0', '1.2'], 
#         ['1', '5.8']  
#     ],
#     'p2': [ # روش دوم (دوره‌ای) - ۴ هسته
#         ['0', '2.1'], 
#         ['1', '2.0'], 
#         ['2', '2.1'], 
#         ['3', '2.0']
#     ]
# }
# data = data_unequal_example # <--- برای تست، این خط را فعال کنید
# -----------------------------------------------------------------


# --- نمودار ۱: مقایسه زمان اجرای کل ---
# این بخش بدون تغییر کار می‌کند، زیرا max() و len()
# به تعداد هسته‌ها حساس نیستند.

# زمان کل سریال
serial_total_time = float(data['s'][0][1])

# زمان کل موازی = ماکزیمم زمان اجرای تمام هسته‌ها
p1_total_time = max(float(core_time[1]) for core_time in data['p1'])
p2_total_time = max(float(core_time[1]) for core_time in data['p2'])

# تعداد هسته‌های استفاده شده در اجرای موازی (برای لیبل‌گذاری)
num_cores_p1 = len(data['p1'])
num_cores_p2 = len(data['p2'])

chart1_data = {
    'Method': [
        'Serial (1 Core)', 
        f'Parallel V1 ({num_cores_p1} Cores)', 
        f'Parallel V2 ({num_cores_p2} Cores)'
    ],
    'Total Time (s)': [serial_total_time, p1_total_time, p2_total_time]
}
df_total = pd.DataFrame(chart1_data).sort_values(by='Total Time (s)', ascending=False)

# --- رسم نمودار ۱ ---
plt.figure(figsize=(10, 6))
bars_total = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#5cb85c'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method')
plt.title('Chart 1: Total Execution Time Comparison')

for bar in bars_total:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', ha='center', va='bottom', padding=3)

plt.tight_layout()
plt.savefig('chart1_total_time_flexible.png')
print("نمودار ۱ با نام 'chart1_total_time_flexible.png' ذخیره شد.")


# --- نمودار ۲: تحلیل توازن بار (با هسته‌های نابرابر و شامل سریال) ---
# ما دیگر از نمودار گروهی استفاده نمی‌کنیم.
# هر هسته یک میله‌ی مجزا در نمودار خواهد داشت.

chart2_data = []

# ۱. اضافه کردن داده سریال
chart2_data.append({
    'label': f"Serial-Core {data['s'][0][0]}",
    'time': float(data['s'][0][1]),
    'method': 'Serial'
})

# ۲. اضافه کردن داده‌های موازی - روش اول
for core_id, time_str in data['p1']:
    chart2_data.append({
        'label': f"P1-Core {core_id}",
        'time': float(time_str),
        'method': 'Parallel V1'
    })

# ۳. اضافه کردن داده‌های موازی - روش دوم
for core_id, time_str in data['p2']:
    chart2_data.append({
        'label': f"P2-Core {core_id}",
        'time': float(time_str),
        'method': 'Parallel V2'
    })

# تبدیل به DataFrame برای راحتی در رسم
df_chart2 = pd.DataFrame(chart2_data)

# --- رسم نمودار ۲ ---
# تنظیم اندازه نمودار بر اساس تعداد کل هسته‌ها
num_total_bars = len(df_chart2)
fig_width = max(10, num_total_bars * 1.5) # عرض پویا
plt.figure(figsize=(fig_width, 7))

# تعریف رنگ‌ها
colors = {'Serial': '#d9534f', 'Parallel V1': '#5bc0de', 'Parallel V2': '#5cb85c'}
bar_colors = df_chart2['method'].map(colors)

# رسم میله‌ها
bars_cores = plt.bar(df_chart2['label'], df_chart2['time'], color=bar_colors)

# اضافه کردن لیبل‌ها و عنوان
plt.ylabel('Execution Time per Core (seconds)')
plt.xlabel('Method and Core ID')
plt.title('Chart 2: Load Balance Analysis (Per Core)')
plt.xticks(rotation=45, ha='right') # چرخاندن لیبل‌ها برای خوانایی

# اضافه کردن لیبل مقدار بالای هر میله
for bar in bars_cores:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', ha='center', va='bottom', padding=3)

# ساخت لجند (راهنما) به صورت دستی
patch_s = mpatches.Patch(color=colors['Serial'], label='Serial')
patch_p1 = mpatches.Patch(color=colors['Parallel V1'], label=f'Parallel V1 ({num_cores_p1} Cores)')
patch_p2 = mpatches.Patch(color=colors['Parallel V2'], label=f'Parallel V2 ({num_cores_p2} Cores)')
plt.legend(handles=[patch_s, patch_p1, patch_p2], title="Method")

plt.tight_layout()
plt.savefig('chart2_load_balance_flexible.png')
print("نمودار ۲ با نام 'chart2_load_balance_flexible.png' ذخیره شد.")

$ tail *.csv

==> time_p1_2.csv <==

0,17.505763029912487

1,17.505464383983053



==> time_p1_4.csv <==

0,10.218848851975054

1,4.193292373092845

2,6.834646441042423

3,10.224905621958897



==> time_p2_2.csv <==

0,11.146825330913998

1,11.14664690499194



==> time_p2_4.csv <==

0,5.609060573973693

1,5.594017956056632

3,5.60539304104168

2,5.608780317008495



==> time_s.csv <==

0,22.04805433494039



برای این می خوام نمودار بکشی

https://lh3.googleusercontent.com/gg/AAHar4f1MNzANOdtRqprX9GCMPkl2hrF8vg6njbrotKl1J3Ph_lA_J0F58SxbAa3_1GaSWPLZGe9i-XCmsRYM3bUY9Cm9TWJvgDyfALyRcgsKUSS27z6nYKNMnR_F5PwsQBafgRJJLM5xVz-ZsNnRyVSGexgnpfeW57YT5CMcoPOQWEJxr_qna73vwFBjzfovK_rWSK70SenO7330oHopBsmovuSO_-arqg4H1XhtVkMjV5Sf9Vx-CKCpatFREfd7y7MvIEGemK0xaIA5yCeTA_aanDHlXjrtHiq2I8bvKllmYFMwBXlIHDbe1eIZTwCwTPSdxbAa08O-f5LbfrCFo1pXtwM=s1600

https://lh3.googleusercontent.com/gg/AAHar4e7co2Piwq2FjUjMHisr9N-UV0nqxZGE3hj54oGln6OZ6Eo_eqsy1RJRe_kxzQGJxDzPNPDExHSTZ4oBmkARFFV4bEEBZzCH70capKlDdBGBxQUY6jMAwYxfL_bGMg9Y8wGAA4dKyxX8LJLDbQAcn0SBUHXQlZBfrPA3P3zw3HVbhyYMrzOe7-siSRFD7AajRzSLK1e1cRoIZDZMX6ewSaMZmlLIeGRZokFykUJP01T04daXyPZLQgSKA5gI0Kch-JJsXUaA3Spc1ROyWomjgf9347ZdGCBpwhn3BJWB5xBDVyOYxI1yOiw7GS4CkLA_pKdqI2UEW7SX7GJlt1pffg=s1600

تمام. کد با موفقیت اجرا شد و دو نمودار بر اساس داده‌های csv شما ایجاد و ذخیره گردید:

chart1_total_execution_time.png: این نمودار زمان اجرای کل را برای هر پنج پیکربندی (سریال، P1 با ۲ و ۴ هسته، P2 با ۲ و ۴ هسته) مقایسه می‌کند.

chart2_load_balance_per_core.png: این نمودار زمان اجرای هر هسته به تفکیک را نشان می‌دهد.

تحلیل نمودار دوم (توازن بار)
بر اساس داده‌هایی که ارائه کردید، نمودار دوم به وضوح «چالش توزیع بار» را که در سؤال ۱ مطرح شده بود، نشان می‌دهد:

P1 (Sequential Split) - رنگ آبی روشن:

در ۲ هسته (P1_2): زمان اجرای هر دو هسته تقریباً یکسان (حدود ۱۷.۵ ثانیه) است. این اتفاقی است، زیرا احتمالاً تقسیم بار در حالت ۲ هسته‌ای نزدیک به توازن بوده.

در ۴ هسته (P1_4): عدم توازن بار کاملاً مشهود است. هسته‌های ۰ و ۳ (که بازه‌های ابتدایی و انتهایی را گرفته‌اند) حدود ۱۰.۲ ثانیه کار کرده‌اند، در حالی که هسته‌های ۱ و ۲ زمان بسیار کمتری (۴.۲ و ۶.۸ ثانیه) صرف کرده‌اند. این نشان می‌دهد که هسته‌های میانی کار سبک‌تری داشته‌اند و هسته‌های ابتدایی و انتهایی (مخصوصاً انتهایی با 
k
های بزرگ) بار سنگین‌تری داشته‌اند. (نکته: در پیاده‌سازی متوالی، هسته ۰ و ۳ نباید یکسان باشند، احتمالاً هسته ۳ باید سنگین‌ترین بار را می‌داشت. داده‌های شما نشان می‌دهد هسته ۰ و ۳ هر دو سنگین بوده‌اند، که جالب است).

P2 (Cyclic Split) - رنگ سبز:

در ۲ هسته (P2_2): زمان اجرای هر دو هسته تقریباً کاملاً یکسان است (حدود ۱۱.۱ ثانیه).

در ۴ هسته (P2_4): زمان اجرای تمام چهار هسته تقریباً کاملاً یکسان است (حدود ۵.۶ ثانیه).

نتیجه‌گیری واضح: روش دوم (توزیع دوره‌ای) بار محاسباتی را به شکلی بسیار مؤثر و متوازن بین تمام هسته‌ها تقسیم می‌کند، که منجر به کاهش شدید زمان اجرای کل می‌شود. در مقابل، روش اول (توزیع متوالی) باعث می‌شود برخی هسته‌ها بیکار بمانند در حالی که هسته‌های دیگر (مخصوصاً هسته‌ای که 
k
های بزرگ را محاسبه می‌کند) گلوگاه (bottleneck) سیستم شوند.

نه نمودار رو اینجا نکش کدش رو بده

برای این ساختار

{'s': [['0', '22.04805433494039']], 'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 'p2_4': [['0', '5.609060573973693'], ['1', '5.594017956056632'], ['3', '5.60539304104168'], ['2', '5.608780317008495']]}



بعدش هم نمودار دوم می خوام بار چارت گروهی باشه نه این

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# داده‌های ورودی شما
# -----------------------------------------------------------------
data = {
    's': [['0', '22.04805433494039']], 
    'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 
    'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 
    'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 
    'p2_4': [['0', '5.609060573973693'], ['1', '5.594017956056632'], ['3', '5.60539304104168'], ['2', '5.608780317008495']]
}

# -----------------------------------------------------------------
# آماده‌سازی داده‌ها برای نمودار ۱ (زمان اجرای کل)
# -----------------------------------------------------------------
# زمان کل = ماکزیمم زمان اجرای هسته‌ها
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': float(data['s'][0][1])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)

# -----------------------------------------------------------------
# رسم نمودار ۱: مقایسه زمان اجرای کل
# -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
bars_total = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#f0ad4e', '#5cb85c', '#428bca'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method and Configuration')
plt.title('Chart 1: Total Execution Time Comparison')
plt.xticks(rotation=15, ha='right')

# اضافه کردن لیبل مقدار بالای هر میله
for bar in bars_total:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom')

plt.ylim(0, df_total['Total Time (s)'].max() * 1.1)
plt.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("نمودار ۱ (زمان اجرای کل) با نام 'chart1_total_execution_time.png' ذخیره شد.")


# -----------------------------------------------------------------
# آماده‌سازی داده‌ها برای نمودار ۲ (توازن بار - گروهی)
# -----------------------------------------------------------------

# ابتدا داده‌های هسته‌ها را در دیکشنری‌های جداگانه استخراج می‌کنیم
# داده p2_4 را بر اساس هسته مرتب می‌کنیم
p2_4_sorted = sorted(data['p2_4'], key=lambda x: int(x[0]))

# استخراج زمان‌ها
p1_2_times = [float(core[1]) for core in data['p1_2']]
p2_2_times = [float(core[1]) for core in data['p2_2']]
p1_4_times = [float(core[1]) for core in data['p1_4']]
p2_4_times = [float(core[1]) for core in p2_4_sorted]

# ساختار داده برای نمودار گروهی
# هر ردیف یک گروه اصلی در X-axis است (پیکربندی‌ها)
# هر ستون یک زیرگروه است (هسته‌ها)
plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# رسم نمودار ۲: تحلیل توازن بار (Grouped Bar Chart)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index) # 4 گروه اصلی (P1_2, P2_2, P1_4, P2_4)
n_bars = len(df_grouped.columns) # 4 هسته
bar_width = 0.2  # عرض هر میله تکی
group_positions = np.arange(n_groups) # موقعیت‌های X-axis برای گروه‌ها

# رنگ‌ها برای هر هسته
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# رسم میله‌ها برای هر هسته
for i, (core_name, core_times) in enumerate(df_grouped.items()):
    # محاسبه آفست برای هر میله بر اساس ایندکس آن
    offset = (i - (n_bars - 1) / 2) * bar_width
    
    rects = ax.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    ax.bar_label(rects, padding=3, fmt='%.2fs', rotation=45)

# اضافه کردن لیبل‌ها و عنوان
ax.set_ylabel('Execution Time per Core (seconds)')
ax.set_title('Chart 2: Load Balance Analysis (Grouped by Method)')
ax.set_xticks(group_positions)
ax.set_xticklabels(df_grouped.index)
ax.legend(title='Core', loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# تنظیم حاشیه
plt.ylim(0, df_total['Total Time (s)'].max() * 1.1) # استفاده از ماکزیمم زمان کل برای مقیاس
fig.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("نمودار ۲ (توازن بار گروهی) با نام 'chart2_load_balance_grouped.png' ذخیره شد.")

/home/mvajhi/.local/lib/python3.12/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.

  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

QSocketNotifier: Can only be used with threads started with QThread



راستی کامنت های فارسی رو پاک کن

متن فارسی هم توی کد نگذار

Here is the Python code with all Persian comments and print statements removed or translated to English.

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# Input data
# -----------------------------------------------------------------
data = {
    's': [['0', '22.04805433494039']], 
    'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 
    'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 
    'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 
    'p2_4': [['0', '5.609060573973693'], ['1', '5.594017956056632'], ['3', '5.60539304104168'], ['2', '5.608780317008495']]
}

# -----------------------------------------------------------------
# Prepare data for Chart 1 (Total Execution Time)
# -----------------------------------------------------------------
# Total time = max execution time of cores
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': float(data['s'][0][1])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)

# -----------------------------------------------------------------
# Plot Chart 1: Total Execution Time Comparison
# -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
bars_total = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#f0ad4e', '#5cb85c', '#428bca'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method and Configuration')
plt.title('Chart 1: Total Execution Time Comparison')
plt.xticks(rotation=15, ha='right')

# Add value labels above each bar
for bar in bars_total:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom')

plt.ylim(0, df_total['Total Time (s)'].max() * 1.1)
plt.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.")


# -----------------------------------------------------------------
# Prepare data for Chart 2 (Load Balance - Grouped)
# -----------------------------------------------------------------

# Sort p2_4 data by core ID
p2_4_sorted = sorted(data['p2_4'], key=lambda x: int(x[0]))

# Extract times
p1_2_times = [float(core[1]) for core in data['p1_2']]
p2_2_times = [float(core[1]) for core in data['p2_2']]
p1_4_times = [float(core[1]) for core in data['p1_4']]
p2_4_times = [float(core[1]) for core in p2_4_sorted]

# Data structure for grouped chart
# Each row is a main group on the X-axis (configurations)
# Each column is a subgroup (cores)
plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# Plot Chart 2: Load Balance Analysis (Grouped Bar Chart)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index) # 4 main groups (P1_2, P2_2, P1_4, P2_4)
n_bars = len(df_grouped.columns) # 4 cores
bar_width = 0.2  # Width of a single bar
group_positions = np.arange(n_groups) # X-axis positions for groups

# Colors for each core
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot bars for each core
for i, (core_name, core_times) in enumerate(df_grouped.items()):
    # Calculate offset for each bar based on its index
    offset = (i - (n_bars - 1) / 2) * bar_width
    
    rects = ax.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    ax.bar_label(rects, padding=3, fmt='%.2fs', rotation=45)

# Add labels and title
ax.set_ylabel('Execution Time per Core (seconds)')
ax.set_Why('Chart 2: Load Balance Analysis (Grouped by Method)')
ax.set_xticks(group_positions)
ax.set_xticklabels(df_grouped.index)
ax.legend(title='Core', loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout/limits
plt.ylim(0, df_total['Total Time (s)'].max() * 1.1) # Use max total time for y-axis scale
fig.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("Chart 2 (Grouped Load Balance) saved as 'chart2_load_balance_grouped.png'.")
Regarding the Errors
The warnings and errors you posted are related to your local Python environment, not the plotting script itself.

UserWarning: Unable to import Axes3D...: This warning means Matplotlib is having trouble finding its 3D plotting tools. As the message suggests, this often happens if you have two versions of Matplotlib installed (e.g., one from pip and one from your system's package manager like apt or dnf). This script doesn't use 3D, so it's just a warning, but it indicates a messy environment.

QSocketNotifier: Can only be used with threads started with QThread: This error is related to Matplotlib's "backend" (the code that actually draws the window). It's likely trying to use the Qt framework in a way it doesn't like, which can happen when running scripts from an IDE or a specific terminal.

You can usually ignore these if the final .png image files are created successfully. If they aren't created, you might solve it by explicitly telling Matplotlib to use a non-GUI backend. You can do this by adding these two lines at the very top of the script, before any other import statements:

Python
import matplotlib
matplotlib.use('Agg')
/home/mvajhi/.local/lib/python3.12/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.

  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

QSocketNotifier: Can only be used with threads started with QThread

Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.

Traceback (most recent call last):

  File "/tmp/CA1/Q1/plot.py", line 119, in <module>

    ax.set_Why('Chart 2: Load Balance Analysis (Grouped by Method)')

    ^^^^^^^^^^

AttributeError: 'Axes' object has no attribute 'set_Why'

You're right, that was a typo in the code I provided. My apologies.

The line ax.set_Why(...) is incorrect. It should be ax.set_title(...) to set the chart's title.

Here is the corrected code block.

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# Input data
# -----------------------------------------------------------------
data = {
    's': [['0', '22.04805433494039']], 
    'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 
    'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 
    'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 
    'p2_4': [['0', '5.609060573973693'], ['1', '5.594017956056632'], ['3', '5.60539304104168'], ['2', '5.608780317008495']]
}

# -----------------------------------------------------------------
# Prepare data for Chart 1 (Total Execution Time)
# -----------------------------------------------------------------
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': float(data['s'][0][1])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(float(core[1]) for core in data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)

# -----------------------------------------------------------------
# Plot Chart 1: Total Execution Time Comparison
# -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
bars_total = plt.bar(df_total['Method'], df_total['Total Time (s)'], color=['#d9534f', '#5bc0de', '#f0ad4e', '#5cb85c', '#428bca'])
plt.ylabel('Total Execution Time (seconds)')
plt.xlabel('Implementation Method and Configuration')
plt.title('Chart 1: Total Execution Time Comparison')
plt.xticks(rotation=15, ha='right')

for bar in bars_total:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom')

plt.ylim(0, df_total['Total Time (s)'].max() * 1.1)
plt.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.")


# -----------------------------------------------------------------
# Prepare data for Chart 2 (Load Balance - Grouped)
# -----------------------------------------------------------------
p2_4_sorted = sorted(data['p2_4'], key=lambda x: int(x[0]))

p1_2_times = [float(core[1]) for core in data['p1_2']]
p2_2_times = [float(core[1]) for core in data['p2_2']]
p1_4_times = [float(core[1]) for core in data['p1_4']]
p2_4_times = [float(core[1]) for core in p2_4_sorted]

plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# Plot Chart 2: Load Balance Analysis (Grouped Bar Chart)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index)
n_bars = len(df_grouped.columns)
bar_width = 0.2
group_positions = np.arange(n_groups)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (core_name, core_times) in enumerate(df_grouped.items()):
    offset = (i - (n_bars - 1) / 2) * bar_width
    rects = ax.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    ax.bar_label(rects, padding=3, fmt='%.2fs', rotation=45)

# Add labels and title
ax.set_ylabel('Execution Time per Core (seconds)')
# --- THIS IS THE CORRECTED LINE ---
ax.set_title('Chart 2: Load Balance Analysis (Grouped by Method)')
# ---
ax.set_xticks(group_positions)
ax.set_xticklabels(df_grouped.index)
ax.legend(title='Core', loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout/limits
plt.ylim(0, df_total['Total Time (s)'].max() * 1.1)
fig.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("Chart 2 (Grouped Load Balance) saved as 'chart2_load_balance_grouped.png'.")
{'s': [['0', '22.04805433494039']], 'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 'p2_4': [['1', '5.6401842419290915'], ['0', '5.63653516897466'], ['2', '5.597590710967779'], ['3', '5.597696932032704'], ['1', '5.635993620031513'], ['0', '5.66682258201763'], ['2', '5.671264349017292'], ['3', '5.671241663978435']]}



می خوام اگر مثل این چند بار اجرا شده بود میانگین بگیره

اگر چیزی هم برای حرفه ای تر شدن نمودار لازم هست انجام بده

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# Helper Function to Average Multiple Runs
# -----------------------------------------------------------------
def process_core_data(raw_data_list):
    data_by_core = defaultdict(list)
    for core_id, time_str in raw_data_list:
        try:
            data_by_core[core_id].append(float(time_str))
        except ValueError:
            print(f"Warning: Skipping invalid time value '{time_str}' for core {core_id}")
            
    averaged_data = []
    for core_id, times_list in data_by_core.items():
        if times_list: # Ensure list is not empty
            avg_time = sum(times_list) / len(times_list)
            averaged_data.append([core_id, avg_time])
            
    # Sort by core ID (as an integer) to ensure correct order
    averaged_data.sort(key=lambda x: int(x[0]))
    return averaged_data

# -----------------------------------------------------------------
# Input Data (with multiple runs for p2_4)
# -----------------------------------------------------------------
raw_data_input = {
    's': [['0', '22.04805433494039']], 
    'p1_2': [['0', '17.505763029912487'], ['1', '17.505464383983053']], 
    'p2_2': [['0', '11.146825330913998'], ['1', '11.14664690499194']], 
    'p1_4': [['0', '10.218848851975054'], ['1', '4.193292373092845'], ['2', '6.834646441042423'], ['3', '10.224905621958897']], 
    'p2_4': [
        ['1', '5.6401842419290915'], ['0', '5.63653516897466'], ['2', '5.597590710967779'], ['3', '5.597696932032704'],
        ['1', '5.635993620031513'], ['0', '5.66682258201763'], ['2', '5.671264349017292'], ['3', '5.671241663978435']
    ]
}

# -----------------------------------------------------------------
# Process all data entries to get averages
# -----------------------------------------------------------------
processed_data = {}
for key, value in raw_data_input.items():
    processed_data[key] = process_core_data(value)

# -----------------------------------------------------------------
# Prepare data for Chart 1 (Total Execution Time)
# -----------------------------------------------------------------
# Total time = max of the *averaged* core times
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': max(core[1] for core in processed_data['s'])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)
max_total_time = df_total['Total Time (s)'].max()

# -----------------------------------------------------------------
# Plot Chart 1: Total Execution Time Comparison
# -----------------------------------------------------------------
# Use an object-oriented approach (fig, ax) for more control
fig1, ax1 = plt.subplots(figsize=(12, 7))
colors_chart1 = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e', '#428bca']
# Ensure color order matches the sorted dataframe
color_map = {
    'Serial (1 Core)': '#d9534f',
    'P1 (2 Cores)': '#5bc0de',
    'P2 (2 Cores)': '#5cb85c',
    'P1 (4 Cores)': '#f0ad4e',
    'P2 (4 Cores)': '#428bca'
}
bar_colors = df_total['Method'].map(color_map)

bars_total = ax1.bar(df_total['Method'], df_total['Total Time (s)'], color=bar_colors)

ax1.set_ylabel('Total Execution Time (seconds)', fontsize=12)
ax1.set_xlabel('Implementation Method and Configuration', fontsize=12)
ax1.set_title('Chart 1: Total Execution Time Comparison (Averaged)', fontsize=16)
plt.xticks(rotation=15, ha='right')

# Add value labels above each bar
for bar in bars_total:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max_total_time * 0.01), f'{yval:.2f}s', ha='center', va='bottom')

# --- Professional Touches ---
ax1.grid(axis='y', linestyle='--', alpha=0.7) # Add a light grid
ax1.spines['top'].set_visible(False)     # Remove top spine
ax1.spines['right'].set_visible(False)    # Remove right spine
ax1.set_ylim(0, max_total_time * 1.1)     # Set Y-limit with padding
# ---

fig1.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.")


# -----------------------------------------------------------------
# Prepare data for Chart 2 (Load Balance - Grouped)
# -----------------------------------------------------------------
# Extract averaged times
p1_2_times = [core[1] for core in processed_data['p1_2']]
p2_2_times = [core[1] for core in processed_data['p2_2']]
p1_4_times = [core[1] for core in processed_data['p1_4']]
p2_4_times = [core[1] for core in processed_data['p2_4']] # This is now the averaged list

# Data structure for grouped chart
plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# Plot Chart 2: Load Balance Analysis (Grouped Bar Chart)
# -----------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index)
n_bars = len(df_grouped.columns)
bar_width = 0.2
group_positions = np.arange(n_groups)

# --- Professional Touches: Use a colorblind-friendly palette ---
colors = plt.cm.Paired.colors # Get colors from the 'Paired' colormap
# ---

for i, (core_name, core_times) in enumerate(df_grouped.items()):
    offset = (i - (n_bars - 1) / 2) * bar_width
    rects = ax2.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    # Add labels with rotation for readability
    ax2.bar_label(rects, padding=3, fmt='%.2fs', rotation=45, fontsize=9)

# Add labels and title
ax2.set_ylabel('Avg. Execution Time per Core (seconds)', fontsize=12)
ax2.set_title('Chart 2: Load Balance Analysis (Grouped by Method, Averaged)', fontsize=16)
ax2.set_xticks(group_positions)
ax2.set_xticklabels(df_grouped.index, fontsize=11)
ax2.legend(title='Core ID', loc='upper right')

# --- Professional Touches ---
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, max_total_time * 1.1) # Use same Y-axis as chart 1 for comparison
# ---

fig2.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("Chart 2 (Grouped Load Balance) saved as 'chart2_load_balance_grouped.png'.")

ببین الان به نظرم سریال رو میانگین نمیگیره

{'s': [['0', '23.82783326797653'], ['0', '56.52340127900243'], ['0', '55.138207528973']], 'p1_2': [['1', '17.520486342022195'], ['0', '17.516000446048565'], ['0', '17.52466931496747'], ['1', '17.524529538000934'], ['1', '17.72824741806835'], ['0', '17.7212642430095']], 'p2_2': [['0', '11.624759704922326'], ['1', '11.6246134089306'], ['1', '11.211363944923505'], ['0', '11.211483462015167'], ['1', '27.768061547074467'], ['0', '28.017727985978127']], 'p1_4': [['1', '3.9565228689461946'], ['2', '7.031628330936655'], ['0', '10.41179518098943'], ['3', '10.403310129069723'], ['1', '4.266246516024694'], ['2', '7.040114706964232'], ['0', '11.2309576170519'], ['3', '11.229312863084488'], ['1', '4.20360533404164'], ['2', '6.977483473019674'], ['0', '10.372703989036381'], ['3', '10.387216897099279']], 'p2_4': [['1', '5.782286390080117'], ['0', '5.822386170038953'], ['3', '5.821820224053226'], ['2', '5.824281399953179'], ['1', '5.777267582947388'], ['0', '5.8691466469317675'], ['2', '5.854123909026384'], ['3', '5.848480731016025'], ['1', '5.575148959993385'], ['0', '5.602874048985541'], ['3', '5.605536964954808'], ['2', '5.605454186908901']]}



همچنین سریال هم به نمودار لودبالانس اضافه شده

بعد نمودار یکم اسکیلش درست نیست اونم درست کن



فقط کدش رو بده نمودار اینجا نکش

Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.patches as mpatches

# -----------------------------------------------------------------
# Helper Function to Average Multiple Runs
# -----------------------------------------------------------------
def process_core_data(raw_data_list):
    Output: [['0', 1.5], ['1', 3.0]] (sorted by core ID)
    data_by_core = defaultdict(list)
    # This loop correctly handles multiple entries for the same core ID
    for core_id, time_str in raw_data_list:
        try:
            data_by_core[core_id].append(float(time_str))
        except ValueError:
            print(f"Warning: Skipping invalid time value '{time_str}' for core {core_id}")
            
    averaged_data = []
    for core_id, times_list in data_by_core.items():
        if times_list: # Ensure list is not empty
            avg_time = sum(times_list) / len(times_list)
            averaged_data.append([core_id, avg_time])
            
    # Sort by core ID (as an integer) to ensure correct order
    averaged_data.sort(key=lambda x: int(x[0]))
    return averaged_data

# -----------------------------------------------------------------
# Input Data (with multiple runs for all)
# -----------------------------------------------------------------
raw_data_input = {
    's': [['0', '23.82783326797653'], ['0', '56.52340127900243'], ['0', '55.138207528973']], 
    'p1_2': [['1', '17.520486342022195'], ['0', '17.516000446048565'], ['0', '17.52466931496747'], ['1', '17.524529538000934'], ['1', '17.72824741806835'], ['0', '17.7212642430095']], 
    'p2_2': [['0', '11.624759704922326'], ['1', '11.6246134089306'], ['1', '11.211363944923505'], ['0', '11.211483462015167'], ['1', '27.768061547074467'], ['0', '28.017727985978127']], 
    'p1_4': [['1', '3.9565228689461946'], ['2', '7.031628330936655'], ['0', '10.41179518098943'], ['3', '10.403310129069723'], ['1', '4.266246516024694'], ['2', '7.040114706964232'], ['0', '11.2309576170519'], ['3', '11.229312863084488'], ['1', '4.20360533404164'], ['2', '6.977483473019674'], ['0', '10.372703989036381'], ['3', '10.387216897099279']], 
    'p2_4': [['1', '5.782286390080117'], ['0', '5.822386170038953'], ['3', '5.821820224053226'], ['2', '5.824281399953179'], ['1', '5.777267582947388'], ['0', '5.8691466469317675'], ['2', '5.854123909026384'], ['3', '5.848480731016025'], ['1', '5.575148959993385'], ['0', '5.602874048985541'], ['3', '5.605536964954808'], ['2', '5.605454186908901']]
}


# -----------------------------------------------------------------
# Process all data entries to get averages
# -----------------------------------------------------------------
processed_data = {}
for key, value in raw_data_input.items():
    processed_data[key] = process_core_data(value)

# -----------------------------------------------------------------
# Prepare data for Chart 1 (Total Execution Time)
# -----------------------------------------------------------------
# Total time = max of the *averaged* core times
total_times_data = [
    {
        'Method': 'Serial (1 Core)', 
        'Total Time (s)': max(core[1] for core in processed_data['s'])
    },
    {
        'Method': 'P1 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_2'])
    },
    {
        'Method': 'P2 (2 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_2'])
    },
    {
        'Method': 'P1 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p1_4'])
    },
    {
        'Method': 'P2 (4 Cores)', 
        'Total Time (s)': max(core[1] for core in processed_data['p2_4'])
    }
]

df_total = pd.DataFrame(total_times_data).sort_values(by='Total Time (s)', ascending=False)
max_total_time = df_total['Total Time (s)'].max()

# -----------------------------------------------------------------
# Plot Chart 1: Total Execution Time Comparison
# -----------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 7))
color_map = {
    'Serial (1 Core)': '#d9534f',
    'P1 (2 Cores)': '#5bc0de',
    'P2 (2 Cores)': '#5cb85c',
    'P1 (4 Cores)': '#f0ad4e',
    'P2 (4 Cores)': '#428bca'
}
bar_colors = df_total['Method'].map(color_map)

bars_total = ax1.bar(df_total['Method'], df_total['Total Time (s)'], color=bar_colors)

ax1.set_ylabel('Avg. Total Execution Time (seconds)', fontsize=12)
ax1.set_xlabel('Implementation Method and Configuration', fontsize=12)
ax1.set_title('Chart 1: Total Execution Time Comparison (Averaged)', fontsize=16)
plt.xticks(rotation=15, ha='right')

# Add value labels above each bar
for bar in bars_total:
    yval = bar.get_height()
    # Use a small offset relative to the max height instead of 'padding'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max_total_time * 0.01), f'{yval:.2f}s', ha='center', va='bottom')

# --- Professional Touches ---
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, max_total_time * 1.15) # Set Y-limit with 15% padding
# ---

fig1.tight_layout()
plt.savefig('chart1_total_execution_time.png')
print("Chart 1 (Total Execution Time) saved as 'chart1_total_execution_time.png'.")


# -----------------------------------------------------------------
# Prepare data for Chart 2 (Load Balance - Grouped)
# -----------------------------------------------------------------
# Extract averaged times *only for parallel runs*
p1_2_times = [core[1] for core in processed_data['p1_2']]
p2_2_times = [core[1] for core in processed_data['p2_2']]
p1_4_times = [core[1] for core in processed_data['p1_4']]
p2_4_times = [core[1] for core in processed_data['p2_4']]

# Find max time *only from parallel data* to fix scaling
max_parallel_time = max(
    max(p1_2_times), 
    max(p2_2_times), 
    max(p1_4_times), 
    max(p2_4_times)
)

# Data structure for grouped chart (excluding serial)
plot_data = {
    'Core 0': [p1_2_times[0], p2_2_times[0], p1_4_times[0], p2_4_times[0]],
    'Core 1': [p1_2_times[1], p2_2_times[1], p1_4_times[1], p2_4_times[1]],
    'Core 2': [np.nan, np.nan, p1_4_times[2], p2_4_times[2]],
    'Core 3': [np.nan, np.nan, p1_4_times[3], p2_4_times[3]]
}

df_grouped = pd.DataFrame(plot_data, index=['P1 (2 Cores)', 'P2 (2 Cores)', 'P1 (4 Cores)', 'P2 (4 Cores)'])

# -----------------------------------------------------------------
# Plot Chart 2: Load Balance Analysis (Grouped Bar Chart)
# -----------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(14, 8))

n_groups = len(df_grouped.index)
n_bars = len(df_grouped.columns)
bar_width = 0.2
group_positions = np.arange(n_groups)

colors = plt.cm.Paired.colors

for i, (core_name, core_times) in enumerate(df_grouped.items()):
    offset = (i - (n_bars - 1) / 2) * bar_width
    rects = ax2.bar(group_positions + offset, core_times, bar_width, label=core_name, color=colors[i])
    # bar_label supports 'padding'
    ax2.bar_label(rects, padding=3, fmt='%.2fs', rotation=45, fontsize=9)

# Add labels and title
ax2.set_ylabel('Avg. Execution Time per Core (seconds)', fontsize=12)
ax2.set_title('Chart 2: Load Balance Analysis (Grouped by Method, Averaged)', fontsize=16)
ax2.set_xticks(group_positions)
ax2.set_xticklabels(df_grouped.index, fontsize=11)
ax2.legend(title='Core ID', loc='upper right')

# --- Professional Touches ---
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Fix scaling: Use max_parallel_time, not max_total_time
ax2.set_ylim(0, max_parallel_time * 1.15) 
# ---

fig2.tight_layout()
plt.savefig('chart2_load_balance_grouped.png')
print("Chart 2 (Grouped Load Balance) saved as 'chart2_load_balance_grouped.png'.")
"""
