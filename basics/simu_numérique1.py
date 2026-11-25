"""
Circuit RC - Interface Moderne avec Tkinter et CustomTkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time

# Configuration du thème
ctk.set_appearance_mode("light")  # "dark" ou "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class CircuitRC:
    """Classe pour modéliser un circuit RC"""
    
    def __init__(self, R, C, E, V0=0):
        self.R = R
        self.C = C * 1e-6  # Conversion µF -> F
        self.E = E
        self.V0 = V0
        self.tau = self.R * self.C
    
    def solution_analytique(self, t):
        """Solution analytique"""
        return self.E * (1 - np.exp(-t / self.tau)) + self.V0 * np.exp(-t / self.tau)
    
    def derivee(self, t, v):
        """Fonction dérivée"""
        return (self.E - v) / self.tau
    
    def methode_euler(self, t_max, dt):
        """Méthode d'Euler"""
        n_points = int(t_max / dt) + 1
        t = np.linspace(0, t_max, n_points)
        v = np.zeros(n_points)
        v[0] = self.V0
        
        for i in range(n_points - 1):
            v[i + 1] = v[i] + dt * self.derivee(t[i], v[i])
        
        return t, v
    
    def methode_rk4(self, t_max, dt):
        """Méthode Runge-Kutta 4"""
        n_points = int(t_max / dt) + 1
        t = np.linspace(0, t_max, n_points)
        v = np.zeros(n_points)
        v[0] = self.V0
        
        for i in range(n_points - 1):
            k1 = dt * self.derivee(t[i], v[i])
            k2 = dt * self.derivee(t[i] + dt/2, v[i] + k1/2)
            k3 = dt * self.derivee(t[i] + dt/2, v[i] + k2/2)
            k4 = dt * self.derivee(t[i] + dt, v[i] + k3)
            
            v[i + 1] = v[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t, v
    
    def calculer_erreur_rms(self, v_numerique, v_analytique):
        """Calcul de l'erreur RMS"""
        return np.sqrt(np.mean((v_numerique - v_analytique)**2))

class ModernCircuitGUI:
    """Interface graphique moderne avec CustomTkinter"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("⚡ Circuit RC - Simulation Moderne")
        self.root.geometry("1400x800")
        
        # Variables
        self.R = 1000
        self.C = 10
        self.E = 5
        self.V0 = 0
        self.methode_active = "all"
        self.animation_active = False
        
        self.setup_ui()
        self.circuit = CircuitRC(self.R, self.C, self.E, self.V0)
        self.update_chart()
    
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        
        # Configuration de la grille principale
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # === PANNEAU LATÉRAL ===
        self.sidebar = ctk.CTkFrame(self.root, width=350, corner_radius=20)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.sidebar.grid_propagate(False)
        
        # Titre
        title = ctk.CTkLabel(
            self.sidebar, 
            text="⚡ Circuit RC", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=(30, 40))
        
        # Contrôles des paramètres
        self.create_parameter_controls()
        
        # Sélecteur de méthode
        self.create_method_selector()
        
        # Panneau d'informations
        self.create_info_panel()
        
        # === ZONE PRINCIPALE ===
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=20)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # En-tête avec titre et boutons
        self.create_header()
        
        # Zone graphique
        self.create_chart_area()
    
    def create_parameter_controls(self):
        """Création des contrôles de paramètres"""
        
        # Résistance
        ctk.CTkLabel(
            self.sidebar, 
            text="Résistance R (Ω)", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 5))
        
        self.R_slider = ctk.CTkSlider(
            self.sidebar, 
            from_=100, 
            to=10000, 
            number_of_steps=99,
            command=self.on_R_change
        )
        self.R_slider.set(1000)
        self.R_slider.pack(pady=5, padx=20, fill="x")
        
        self.R_label = ctk.CTkLabel(self.sidebar, text="1000 Ω")
        self.R_label.pack()
        
        # Capacité
        ctk.CTkLabel(
            self.sidebar, 
            text="Capacité C (µF)", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 5))
        
        self.C_slider = ctk.CTkSlider(
            self.sidebar, 
            from_=1, 
            to=100, 
            number_of_steps=99,
            command=self.on_C_change
        )
        self.C_slider.set(10)
        self.C_slider.pack(pady=5, padx=20, fill="x")
        
        self.C_label = ctk.CTkLabel(self.sidebar, text="10 µF")
        self.C_label.pack()
        
        # Tension
        ctk.CTkLabel(
            self.sidebar, 
            text="Tension E (V)", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 5))
        
        self.E_slider = ctk.CTkSlider(
            self.sidebar, 
            from_=1, 
            to=12, 
            number_of_steps=110,
            command=self.on_E_change
        )
        self.E_slider.set(5)
        self.E_slider.pack(pady=5, padx=20, fill="x")
        
        self.E_label = ctk.CTkLabel(self.sidebar, text="5.0 V")
        self.E_label.pack()
        
        # Tension initiale
        ctk.CTkLabel(
            self.sidebar, 
            text="Tension initiale V₀ (V)", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 5))
        
        self.V0_slider = ctk.CTkSlider(
            self.sidebar, 
            from_=0, 
            to=12, 
            number_of_steps=120,
            command=self.on_V0_change
        )
        self.V0_slider.set(0)
        self.V0_slider.pack(pady=5, padx=20, fill="x")
        
        self.V0_label = ctk.CTkLabel(self.sidebar, text="0.0 V")
        self.V0_label.pack()
    
    def create_method_selector(self):
        """Création du sélecteur de méthode"""
        ctk.CTkLabel(
            self.sidebar, 
            text="Méthode d'affichage", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(30, 10))
        
        self.method_var = ctk.StringVar(value="all")
        
        methods = [
            ("Toutes", "all"),
            ("Analytique", "analytical"),
            ("Euler", "euler"),
            ("RK4", "rk4")
        ]
        
        for text, value in methods:
            radio = ctk.CTkRadioButton(
                self.sidebar,
                text=text,
                variable=self.method_var,
                value=value, 
                command=self.on_method_change
            )
            radio.pack(pady=5)
    
    def create_info_panel(self):
        """Création du panneau d'informations"""
        info_frame = ctk.CTkFrame(self.sidebar, corner_radius=15)
        info_frame.pack(pady=30, padx=20, fill="x")
        
        ctk.CTkLabel(
            info_frame, 
            text="📊 Caractéristiques", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        self.info_labels = {}
        info_items = [
            ("tau", "Constante τ:"),
            ("t63", "t(63%):"),
            ("t95", "t(95%):"),
            ("error_euler", "Erreur Euler:"),
            ("error_rk4", "Erreur RK4:")
        ]
        
        for key, label in info_items:
            frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                frame, 
                text=label, 
                font=ctk.CTkFont(size=12)
            ).pack(side="left")
            
            self.info_labels[key] = ctk.CTkLabel(
                frame, 
                text="--", 
                font=ctk.CTkFont(size=12, weight="bold")
            )
            self.info_labels[key].pack(side="right")
        
        # Espace en bas
        ctk.CTkLabel(info_frame, text="").pack(pady=10)
    
    def create_header(self):
        """Création de l'en-tête"""
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        # Titre
        ctk.CTkLabel(
            header_frame, 
            text="Charge du Condensateur", 
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left")
        
        # Boutons de contrôle
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.pack(side="right")
        
        self.animate_btn = ctk.CTkButton(
            button_frame,
            text="🎬 Animer",
            width=120,
            command=self.toggle_animation
        )
        self.animate_btn.pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="💾 Exporter",
            width=120,
            command=self.export_data
        ).pack(side="left", padx=5)
    
    def create_chart_area(self):
        """Création de la zone graphique"""
        chart_frame = ctk.CTkFrame(self.main_frame, corner_radius=15)
        chart_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        # Configuration matplotlib pour un style moderne
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        
        # Style moderne pour le graphique
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#cccccc')
        self.ax.spines['bottom'].set_color('#cccccc')
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
    
    # Callbacks pour les sliders
    def on_R_change(self, value):
        self.R = int(value)
        self.R_label.configure(text=f"{self.R} Ω")
        self.update_circuit()
    
    def on_C_change(self, value):
        self.C = round(value, 1)
        self.C_label.configure(text=f"{self.C} µF")
        self.update_circuit()
    
    def on_E_change(self, value):
        self.E = round(value, 1)
        self.E_label.configure(text=f"{self.E} V")
        self.update_circuit()
    
    def on_V0_change(self, value):
        self.V0 = round(value, 1)
        self.V0_label.configure(text=f"{self.V0} V")
        self.update_circuit()
    
    def on_method_change(self):
        self.methode_active = self.method_var.get()
        self.update_chart()
    
    def update_circuit(self):
        """Mise à jour du circuit et des informations"""
        self.circuit = CircuitRC(self.R, self.C, self.E, self.V0)
        
        # Mise à jour des informations
        tau_ms = self.circuit.tau * 1000
        t63_ms = tau_ms
        t95_ms = tau_ms * np.log(20)
        
        self.info_labels["tau"].configure(text=f"{tau_ms:.2f} ms")
        self.info_labels["t63"].configure(text=f"{t63_ms:.2f} ms")
        self.info_labels["t95"].configure(text=f"{t95_ms:.2f} ms")
        
        self.update_chart()
    
    def update_chart(self):
        """Mise à jour du graphique"""
        t_max = max(self.circuit.tau * 5, 0.01)
        dt = t_max / 1000
        
        # Solutions
        t_anal = np.linspace(0, t_max, 1000)
        v_anal = self.circuit.solution_analytique(t_anal)
        
        t_euler, v_euler = self.circuit.methode_euler(t_max, dt)
        t_rk4, v_rk4 = self.circuit.methode_rk4(t_max, dt)
        
        # Calcul des erreurs
        v_anal_euler = self.circuit.solution_analytique(t_euler)
        v_anal_rk4 = self.circuit.solution_analytique(t_rk4)
        error_euler = self.circuit.calculer_erreur_rms(v_euler, v_anal_euler)
        error_rk4 = self.circuit.calculer_erreur_rms(v_rk4, v_anal_rk4)
        
        self.info_labels["error_euler"].configure(text=f"{error_euler:.2e}")
        self.info_labels["error_rk4"].configure(text=f"{error_rk4:.2e}")
        
        # Nettoyage du graphique
        self.ax.clear()
        
        # Tracés selon la méthode
        if self.methode_active in ["all", "analytical"]:
            self.ax.plot(t_anal * 1000, v_anal, 'r-', linewidth=3, 
                        label='Solution Analytique', alpha=0.8)
        
        if self.methode_active in ["all", "euler"]:
            self.ax.plot(t_euler * 1000, v_euler, 'o', color='#3498db', 
                        markersize=3, label=f'Euler (RMS={error_euler:.2e})', alpha=0.7)
        
        if self.methode_active in ["all", "rk4"]:
            self.ax.plot(t_rk4 * 1000, v_rk4, '^', color='#2ecc71', 
                        markersize=2, label=f'RK4 (RMS={error_rk4:.2e})', alpha=0.7)
        
        # Lignes de référence
        self.ax.axhline(y=self.circuit.E, color='gray', linestyle='--', 
                       alpha=0.5, label=f'E = {self.circuit.E}V')
        self.ax.axvline(x=self.circuit.tau * 1000, color='#f39c12', 
                       linestyle='--', alpha=0.7, 
                       label=f'τ = {self.circuit.tau*1000:.1f}ms')
        
        # Configuration
        self.ax.set_xlabel('Temps (ms)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Tension (V)', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Circuit RC - τ = {self.circuit.tau*1000:.1f} ms', 
                         fontsize=14, fontweight='bold', pad=20)
        self.ax.legend(frameon=True, fancybox=True, shadow=True)
        self.ax.grid(True, alpha=0.3)
        
        # Style moderne
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.canvas.draw()
    
    def toggle_animation(self):
        """Basculer l'animation"""
        if self.animation_active:
            self.stop_animation()
        else:
            self.start_animation()
    
    def start_animation(self):
        """Démarrer l'animation"""
        self.animation_active = True
        self.animate_btn.configure(text="⏹️ Arrêter")
        
        def animate():
            t_max = max(self.circuit.tau * 5, 0.01)
            dt = t_max / 200
            
            t_points = np.arange(0, t_max, dt)
            
            for i, t_current in enumerate(t_points):
                if not self.animation_active:
                    break
                
                t_partial = t_points[:i+1]
                v_partial = self.circuit.solution_analytique(t_partial)
                
                self.ax.clear()
                self.ax.plot(t_partial * 1000, v_partial, 'r-', linewidth=3)
                
                if i > 0:
                    self.ax.plot(t_current * 1000, 
                               self.circuit.solution_analytique(t_current), 
                               'o', color='#f39c12', markersize=10)
                
                self.ax.set_xlim(0, t_max * 1000)
                self.ax.set_ylim(0, self.circuit.E * 1.1)
                self.ax.set_xlabel('Temps (ms)')
                self.ax.set_ylabel('Tension (V)')
                self.ax.set_title('Animation - Charge du Condensateur')
                self.ax.grid(True, alpha=0.3)
                
                self.canvas.draw()
                time.sleep(0.05)
            
            self.stop_animation()
        
        # Lancer dans un thread séparé
        threading.Thread(target=animate, daemon=True).start()
    
    def stop_animation(self):
        """Arrêter l'animation"""
        self.animation_active = False
        self.animate_btn.configure(text="🎬 Animer")
        self.update_chart()
    
    def export_data(self):
        """Exporter les données"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
            )
            
            if filename:
                t_max = max(self.circuit.tau * 5, 0.01)
                dt = t_max / 1000
                
                t_euler, v_euler = self.circuit.methode_euler(t_max, dt)
                t_rk4, v_rk4 = self.circuit.methode_rk4(t_max, dt)
                v_anal = self.circuit.solution_analytique(t_euler)
                
                with open(filename, 'w') as f:
                    f.write(f"# Circuit RC - R={self.R}Ω, C={self.C}µF, E={self.E}V\n")
                    f.write("Temps(ms),Analytique(V),Euler(V),RK4(V)\n")
                    
                    for i in range(len(t_euler)):
                        t_ms = t_euler[i] * 1000
                        rk4_val = v_rk4[i] if i < len(v_rk4) else ""
                        f.write(f"{t_ms:.4f},{v_anal[i]:.6f},{v_euler[i]:.6f},{rk4_val}\n")
                
                messagebox.showinfo("Succès", f"Données exportées vers {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'export: {e}")
    
    def run(self):
        """Lancer l'application"""
        self.root.mainloop()

def main():
    """Fonction principale"""
    try:
        app = ModernCircuitGUI()
        app.run()
    except ImportError:
        print("CustomTkinter n'est pas installé. Installation...")
        print("Exécutez: pip install customtkinter")
        
        # Version fallback avec tkinter standard mais style moderne
        from tkinter import messagebox
        messagebox.showinfo("Information", 
                          "Pour une interface moderne, installez CustomTkinter:\n"
                          "pip install customtkinter\n\n"
                          "Sinon, utilisez la version web HTML.")

if __name__ == "__main__":
    main()