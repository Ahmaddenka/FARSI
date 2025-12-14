import pandas as pd

def build_control_summary(
    xlsx_path: str,
    sheet_name: str | None = None,
    mechanism_col: str | None = None,
    agent_col: str | None = None,
    control_col: str | None = None
) -> dict:
    """
    از فایل اکسل یک دیکشنری می‌سازد که برای هر مکانیزم،
    تعداد عامل (agents) و تعداد کنترل (controls) را نگه می‌دارد.

    خروجی: {mechanism: {"agent_count": int, "control_count": int}}
    """
    # اکسل را بخوان
    if sheet_name is None:
        xls = pd.ExcelFile(xlsx_path)
        # اگر فقط یک شیت داشت همان را بگیر، وگرنه Sheet1/Sheet2 را ترجیح بده
        candidates = ["Sheet1", "Sheet2"]
        sheet_name = next((s for s in candidates if s in xls.sheet_names), xls.sheet_names[0])

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # اگر نام ستون‌ها داده نشده، حدس بزن
    cols_lower = {c: str(c).strip().lower() for c in df.columns}

    if mechanism_col is None:
        # پیش‌فرض: ستون اول مکانیزم است
        mechanism_col = df.columns[0]

    if agent_col is None:
        # هر ستونی که کلمه agent در نامش باشد
        agent_col = next((c for c, lc in cols_lower.items() if "agent" in lc), None)

    if control_col is None:
        # هر ستونی که کلمه control در نامش باشد
        control_col = next((c for c, lc in cols_lower.items() if "control" in lc), None)

    required = [mechanism_col, agent_col, control_col]
    if any(c is None for c in required):
        raise ValueError(
            f"ستون‌های لازم پیدا نشدند. مکانیزم={mechanism_col!r}, agent={agent_col!r}, control={control_col!r}"
        )

    # پاک‌سازی‌های ساده
    df = df[[mechanism_col, agent_col, control_col]].copy()
    df[mechanism_col] = df[mechanism_col].astype(str).str.strip()

    # تبدیل به عدد (اگر مقداری تهی یا متنی باشد صفر می‌گیریم)
    for c in (agent_col, control_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ساخت دیکشنری خروجی
    out = {}
    for _, row in df.iterrows():
        mech = row[mechanism_col]
        out[mech] = {
            "agent_count": int(row[agent_col]),
            "control_count": int(row[control_col]),
        }
    return out

# --- نمونه استفاده ---
# d = build_control_summary(r"/path/to/Control_gorup_structure.xlsx")
# print(d)
