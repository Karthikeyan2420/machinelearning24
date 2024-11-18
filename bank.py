import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3

# Database Setup
def setup_database():
    conn = sqlite3.connect("bank.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            balance REAL DEFAULT 0,
            account_status TEXT DEFAULT 'Active',
            credit_card TEXT DEFAULT 'Not Issued',
            debit_card TEXT DEFAULT 'Not Issued'
        )
    """)
    conn.commit()
    conn.close()

setup_database()

# Centralized Theme Configuration
class Theme:
    BG_COLOR = "#f0f0f0"
    PRIMARY_COLOR = "#2e86c1"
    BUTTON_COLOR = "#1abc9c"
    TEXT_COLOR = "#2c3e50"
    FONT = ("Arial", 12)
    TITLE_FONT = ("Arial", 18, "bold")

# Main Application Window
def main_screen():
    def open_admin_login():
        admin_login_window()

    def open_customer_login():
        customer_login_window()

    root = tk.Tk()
    root.title("Bank Management System")
    root.geometry("600x400")
    root.configure(bg=Theme.BG_COLOR)

    ttk.Label(root, text="Bank Management System", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=20)
    ttk.Label(
        root,
        text='"The best way to predict the future is to create it." - APJ Abdul Kalam',
        font=Theme.FONT,
        foreground=Theme.TEXT_COLOR,
        wraplength=500,
        justify="center",
    ).pack(pady=10)

    ttk.Button(root, text="Admin Login", command=open_admin_login, style="TButton").pack(pady=20)
    ttk.Button(root, text="Customer Login", command=open_customer_login, style="TButton").pack()

    # Customizing button style
    s = ttk.Style()
    s.configure("TButton", font=Theme.FONT, background=Theme.BUTTON_COLOR, foreground="white", padding=10)

    root.mainloop()

# Admin Login
def admin_login_window():
    def validate_admin():
        if username_entry.get() == "admin" and password_entry.get() == "admin":
            messagebox.showinfo("Login Successful", "Welcome, Admin!")
            admin_dashboard()
            admin_login.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid Admin Credentials")

    admin_login = tk.Toplevel()
    admin_login.title("Admin Login")
    admin_login.geometry("400x250")
    admin_login.configure(bg=Theme.BG_COLOR)

    ttk.Label(admin_login, text="Admin Login", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=10)
    ttk.Label(admin_login, text="Username:", font=Theme.FONT).pack(pady=5)
    username_entry = ttk.Entry(admin_login, width=30)
    username_entry.pack()
    ttk.Label(admin_login, text="Password:", font=Theme.FONT).pack(pady=5)
    password_entry = ttk.Entry(admin_login, show="*", width=30)
    password_entry.pack()

    ttk.Button(admin_login, text="Login", command=validate_admin, style="TButton").pack(pady=20)

# Admin Dashboard
def admin_dashboard():
    def view_customers():
        conn = sqlite3.connect("bank.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers")
        customers = cursor.fetchall()
        conn.close()

        customer_window = tk.Toplevel()
        customer_window.title("Customers List")
        customer_window.geometry("600x400")
        customer_window.configure(bg=Theme.BG_COLOR)

        ttk.Label(customer_window, text="Customer Details", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=10)
        for customer in customers:
            ttk.Label(customer_window, text=f"ID: {customer[0]} | Username: {customer[1]} | Balance: ₹{customer[3]:.2f}").pack()

    def add_customer():
        def save_customer():
            username = username_entry.get()
            password = password_entry.get()
            balance = balance_entry.get()

            if not username or not password or not balance.isdigit():
                messagebox.showerror("Invalid Input", "Please enter valid details.")
                return

            conn = sqlite3.connect("bank.db")
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO customers (username, password, balance) VALUES (?, ?, ?)",
                               (username, password, float(balance)))
                conn.commit()
                messagebox.showinfo("Success", "Customer added successfully.")
                add_customer_window.destroy()
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Username already exists.")
            conn.close()

        add_customer_window = tk.Toplevel()
        add_customer_window.title("Add Customer")
        add_customer_window.geometry("400x300")
        add_customer_window.configure(bg=Theme.BG_COLOR)

        ttk.Label(add_customer_window, text="Add Customer", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=10)
        ttk.Label(add_customer_window, text="Username:", font=Theme.FONT).pack(pady=5)
        username_entry = ttk.Entry(add_customer_window)
        username_entry.pack()
        ttk.Label(add_customer_window, text="Password:", font=Theme.FONT).pack(pady=5)
        password_entry = ttk.Entry(add_customer_window, show="*")
        password_entry.pack()
        ttk.Label(add_customer_window, text="Initial Balance:", font=Theme.FONT).pack(pady=5)
        balance_entry = ttk.Entry(add_customer_window)
        balance_entry.pack()

        ttk.Button(add_customer_window, text="Save", command=save_customer, style="TButton").pack(pady=20)

    admin = tk.Toplevel()
    admin.title("Admin Dashboard")
    admin.geometry("500x400")
    admin.configure(bg=Theme.BG_COLOR)

    ttk.Label(admin, text="Admin Dashboard", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=20)
    ttk.Button(admin, text="View All Customers", command=view_customers, style="TButton").pack(pady=10)
    ttk.Button(admin, text="Add Customer", command=add_customer, style="TButton").pack(pady=10)

# Customer Login
def customer_login_window():
    def validate_customer():
        username = username_entry.get()
        password = password_entry.get()

        conn = sqlite3.connect("bank.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE username = ? AND password = ?", (username, password))
        customer = cursor.fetchone()
        conn.close()

        if customer:
            messagebox.showinfo("Login Successful", f"Welcome, {username}!")
            customer_dashboard(customer)
            customer_login.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid Credentials")

    customer_login = tk.Toplevel()
    customer_login.title("Customer Login")
    customer_login.geometry("400x250")
    customer_login.configure(bg=Theme.BG_COLOR)

    ttk.Label(customer_login, text="Customer Login", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=10)
    ttk.Label(customer_login, text="Username:", font=Theme.FONT).pack(pady=5)
    username_entry = ttk.Entry(customer_login, width=30)
    username_entry.pack()
    ttk.Label(customer_login, text="Password:", font=Theme.FONT).pack(pady=5)
    password_entry = ttk.Entry(customer_login, show="*", width=30)
    password_entry.pack()

    ttk.Button(customer_login, text="Login", command=validate_customer, style="TButton").pack(pady=20)

# Customer Dashboard
def customer_dashboard(customer):
    dashboard = tk.Toplevel()
    dashboard.title("Customer Dashboard")
    dashboard.geometry("500x400")
    dashboard.configure(bg=Theme.BG_COLOR)

    ttk.Label(dashboard, text=f"Welcome, {customer[1]}", font=Theme.TITLE_FONT, foreground=Theme.PRIMARY_COLOR).pack(pady=20)
    ttk.Label(dashboard, text=f"Balance: ₹{customer[3]:.2f}", font=Theme.FONT).pack(pady=5)
    ttk.Label(dashboard, text=f"Account Status: {customer[4]}", font=Theme.FONT).pack(pady=5)
    ttk.Label(dashboard, text=f"Credit Card: {customer[5]}", font=Theme.FONT).pack(pady=5)
    ttk.Label(dashboard, text=f"Debit Card: {customer[6]}", font=Theme.FONT).pack(pady=5)

# Run Application
main_screen()
