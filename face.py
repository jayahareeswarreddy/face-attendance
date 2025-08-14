import cv2
import pandas as pd
import os
from datetime import datetime
STUDENTS_FILE='registered_students.csv'
ATTENDANCE_FILE='attendance.csv'
if not os.path.exists(STUDENTS_FILE):
    pd.DataFrame(columns=['Reg_No','Name']).to_csv(STUDENTS_FILE,index=False)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=['Reg_No','Name', 'Date']).to_csv(ATTENDANCE_FILE, index=False)
def register_student():
    reg_no = input("Enter Student ID: ").strip()
    name = input("Enter Student Name: ").strip()
    students_df = pd.read_csv(STUDENTS_FILE)
    if reg_no in students_df['Reg_No'].astype(str).values:
        print("You are already registered.")
        return
    new_student = pd.DataFrame([[reg_no, name]], columns=['Reg_No', 'Name'])
    students_df = pd.concat([students_df, new_student], ignore_index=True)
    students_df.to_csv(STUDENTS_FILE, index=False)
    print(f"Student {name} (ID: {reg_no}) registered successfully.")
def mark_attendance():
    students_df = pd.read_csv(STUDENTS_FILE)
    if students_df.empty:
        print("No students registered. Please register first to mark attendance.")
        return
    reg_no = input("Enter your Student ID to mark attendance: ").strip()
    if reg_no not in students_df['Reg_No'].astype(str).values:
        print("You are not registered. Please register first to mark your attendance.")
        return
    print("[INFO] Scanning face...")
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cam.release()
    cv2.destroyAllWindows()
    if len(faces) == 0:
        print("Face is not visible.")
        return
    name = students_df.loc[students_df['Reg_No'].astype(str) == reg_no, 'Name'].values[0]
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
    new_record = pd.DataFrame([[reg_no, name, date_time]], columns=['Reg_No', 'Name', 'Date'])
    attendance_df = pd.concat([attendance_df, new_record], ignore_index=True)
    attendance_df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"Attendance marked for {name} (ID: {reg_no}) at {date_time}")
def view_attendance():
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
    if attendance_df.empty:
        print("--- No attendance records found ---")
    else:
        print("\n--- Attendance Records ---")
        print(attendance_df)
def view_registered_students():
    students_df = pd.read_csv(STUDENTS_FILE)
    if students_df.empty:
        print("--- No registered students found ---")
    else:
        print("\n--- Registered Students ---")
        print(students_df)
def delete_student():
    reg_no = input("Enter Student ID to delete: ").strip()
    students_df = pd.read_csv(STUDENTS_FILE)
    if reg_no not in students_df['Reg_No'].astype(str).values:
        print("Student not found.")
        return
    students_df = students_df[students_df['Reg_No'].astype(str) != reg_no]
    students_df.to_csv(STUDENTS_FILE, index=False)
    print(f"Student ID {reg_no} deleted successfully.")
def main():
    while True:
        print('''
========= FACE ATTENDANCE SYSTEM =========
1. Register New Student
2. Mark Attendance
3. View Attendance Records
4. View Registered Students
5. Delete Student
6. Exit
==========================================
        ''')
        choice = input("Select an option (1-6): ").strip()
        if choice == '1':
            register_student()
        elif choice == '2':
            mark_attendance()
        elif choice == '3':
            view_attendance()
        elif choice == '4':
            view_registered_students()
        elif choice == '5':
            delete_student()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1-6.")

if __name__ == '__main__':
    main()
