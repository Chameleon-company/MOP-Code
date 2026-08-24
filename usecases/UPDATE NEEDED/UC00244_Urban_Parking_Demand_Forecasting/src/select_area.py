import pandas as pd

# Load dataset
df = pd.read_csv("../data/cleaned_parking_with_area.csv")


# Select Area
areas = sorted(df["area"].dropna().unique())

print("Available Areas:")
for i, area in enumerate(areas, start=1):
    print(f"{i}. {area}")

choice = int(input("\nSelect an area by number: "))

if choice < 1 or choice > len(areas):
    print("Invalid selection.")
    exit()

selected_area = areas[choice - 1]


# Select Hour
hour = int(input("Enter hour (0-23): "))

if hour < 0 or hour > 23:
    print("Invalid hour.")
    exit()


# Select Day
print("\nDay of Week:")
print("0 = Monday")
print("1 = Tuesday")
print("2 = Wednesday")
print("3 = Thursday")
print("4 = Friday")
print("5 = Saturday")
print("6 = Sunday")

day = int(input("Enter day (0-6): "))

if day < 0 or day > 6:
    print("Invalid day.")
    exit()


# Filter Data
area_df = df[df["area"] == selected_area]


# Display Summary
print("\nSelection Summary")
print("-----------------")
print(f"Area: {selected_area}")
print(f"Hour: {hour}")
print(f"Day: {day}")
print(f"Rows in area: {len(area_df)}")
print(f"Unique parking bays: {area_df['bay_id'].nunique()}")