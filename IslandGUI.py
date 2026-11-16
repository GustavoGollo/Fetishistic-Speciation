import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import traceback
import os
from IslandAnalysis import run_all


class IslandApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modelo de Ilhas — Especiação")
        # Aumentei a janela para caber todos os parâmetros confortavelmente
        self.geometry("1100x900")
        
        # --- VARIÁVEIS BÁSICAS ---
        self.seeds_var = tk.StringVar(value="1") # Base seed (mudei de "1 2 3" para "1")
        self.n_reps_var = tk.StringVar(value="32") # Campo de Réplicas

        self.n_islands_var = tk.StringVar(value="3")
        self.mig_rate_var = tk.StringVar(value="0.002")    # Corrigido (valor da imagem)
        self.max_gen_var = tk.StringVar(value="2000")      # Corrigido (o que você quer rodar)
        self.start_gen_var = tk.StringVar(value="1950")    # Corrigido (apenas o número final)

        self.n_cores_var = tk.StringVar(value="6")
        self.out_dir_var = tk.StringVar(value="resultados_ilhas")
        
        # --- VARIÁVEIS AVANÇADAS (Da GUI Speciation) ---
        self.loci_var = tk.StringVar(value="16")
        self.states_var = tk.StringVar(value="9")
        self.init_opt_var = tk.StringVar(value="a")
        self.females_var = tk.StringVar(value="4500")
        self.males_var = tk.StringVar(value="4500")
        self.femsort_var = tk.StringVar(value="80")
        self.preselec_var = tk.StringVar(value="30")
        self.offspring_var = tk.StringVar(value="6")
        self.mut_rate_var = tk.StringVar(value="0.005")
        self.dbscan_eps_var = tk.StringVar(value="1.00")
        self.dbscan_min_var = tk.StringVar(value="110")
        self.alpha_var = tk.StringVar(value="0.01")
        
        self._build_ui()

    def _build_ui(self):
        # Frame principal dividido em grid
        main = ttk.Frame(self, padding=10)
        main.pack(fill='both', expand=True)
        
        # Criando os dois lados (Esquerda: Parâmetros | Direita: Log)
        left_frame = ttk.Frame(main)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10)) # Padx separa um do outro
        
        right_frame = ttk.Frame(main)
        right_frame.grid(row=0, column=1, sticky='nsew')
        
        # Configurando o grid para expandir corretamente
        main.columnconfigure(0, weight=1) # A coluna da esquerda cresce
        main.columnconfigure(1, weight=1) # A coluna da direita cresce
        main.rowconfigure(0, weight=1)
        
        def add_row(parent, label, var, row):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
            ttk.Entry(parent, textvariable=var, width=15).grid(row=row, column=1, sticky='w', padx=5)

        # ==========================================
        # LADO ESQUERDO: PARÂMETROS E BOTÕES
        # ==========================================
        frm_params = ttk.LabelFrame(left_frame, text="Parâmetros Básicos", padding=10)
        frm_params.pack(fill='x', pady=5)
        
        add_row(frm_params, "Base Seed:", self.seeds_var, 0)
        add_row(frm_params, "Replicas (spec & neutral):", self.n_reps_var, 1)
        add_row(frm_params, "Number of islands:", self.n_islands_var, 2)
        add_row(frm_params, "Migration rate (0 a 1):", self.mig_rate_var, 3)
        add_row(frm_params, "Max generations:", self.max_gen_var, 4)
        add_row(frm_params, "Colection start:", self.start_gen_var, 5)
        add_row(frm_params, "Processor cores:", self.n_cores_var, 6)
        add_row(frm_params, "Output folder:", self.out_dir_var, 7)
        
        frm_adv = ttk.LabelFrame(left_frame, text="Parâmetros Genéticos", padding=10)
        frm_adv.pack(fill='x', pady=5)
        
        add_row(frm_adv, "Loci count:", self.loci_var, 0)
        add_row(frm_adv, "States count:", self.states_var, 1)
        add_row(frm_adv, "Init option (a-d):", self.init_opt_var, 2)
        add_row(frm_adv, "Females:", self.females_var, 3)
        add_row(frm_adv, "Males:", self.males_var, 4)
        add_row(frm_adv, "FemSort:", self.femsort_var, 5)
        add_row(frm_adv, "PreSelec:", self.preselec_var, 6)
        add_row(frm_adv, "Offspring/female:", self.offspring_var, 7)
        add_row(frm_adv, "Mutation rate:", self.mut_rate_var, 8)
        
        frm_dbscan = ttk.LabelFrame(left_frame, text="Análise e Agrupamento (DBSCAN)", padding=10)
        frm_dbscan.pack(fill='x', pady=5)
        
        add_row(frm_dbscan, "DBSCAN eps:", self.dbscan_eps_var, 0)
        add_row(frm_dbscan, "DBSCAN min_samples:", self.dbscan_min_var, 1)
        add_row(frm_dbscan, "Significance alpha:", self.alpha_var, 2)

        # Botões (na esquerda, embaixo dos parâmetros)
        frm_btn = ttk.Frame(left_frame)
        frm_btn.pack(fill='x', pady=10)
        
        self.btn_run = ttk.Button(frm_btn, text="▶ RUN SIMULATION", command=self._run)
        self.btn_run.pack(side='left', padx=5)
        
        ttk.Button(frm_btn, text="📂 Open folder", command=self._open_folder).pack(side='left', padx=5)
        
        ttk.Button(frm_btn, text="🔄 Reset", command=self._reset_params).pack(side='left', padx=5)

# ==========================================
        # LADO DIREITO: LOG DE EXECUÇÃO
        # ==========================================
        frm_log = ttk.LabelFrame(right_frame, text="Log de Execução", padding=10)
        frm_log.pack(fill='both', expand=True, pady=5)
        
        self.txt_log = tk.Text(frm_log, bg="black", fg="white", state='disabled')
        self.txt_log.pack(fill='both', expand=True, side='left')
        
        scrollbar = ttk.Scrollbar(frm_log, orient="vertical", command=self.txt_log.yview)
        scrollbar.pack(side='right', fill='y')
        self.txt_log.config(yscrollcommand=scrollbar.set)



    def _log(self, message):
        """Adiciona mensagens na tela preta de log de forma segura."""
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END) # Rola para o final automaticamente
        self.txt_log.config(state='disabled')

    def _open_folder(self):
        """Abre a pasta de resultados no Windows Explorer."""
        folder = self.out_dir_var.get()
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            messagebox.showwarning("Aviso", "A pasta ainda não existe.")

    def _reset_params(self):
        """Reseta todos os campos para os valores padrão e limpa o log."""
        # Variáveis básicas
        self.seeds_var.set("1")
        self.n_reps_var.set("32")
        self.n_islands_var.set("3")
        self.mig_rate_var.set("0.002")
        self.max_gen_var.set("2000") 
        self.start_gen_var.set("1950") # Sugestão de start baseada em MaxGen=1000
        self.n_cores_var.set("6")
        self.out_dir_var.set("resultados_ilhas")
        
        # Variáveis avançadas
        self.loci_var.set("16")
        self.states_var.set("9")
        self.init_opt_var.set("a")
        self.females_var.set("4000")
        self.males_var.set("4000")
        self.femsort_var.set("80")
        self.preselec_var.set("30")
        self.offspring_var.set("6")
        self.mut_rate_var.set("0.005")
        
        # Parâmetros do DBSCAN (Usando os que você achou no gráfico do joelho)
        self.dbscan_eps_var.set("0.15")
        self.dbscan_min_var.set("120")
        self.alpha_var.set("0.01")
        
        # Limpa a tela preta de log
        self.txt_log.config(state='normal')
        self.txt_log.delete(1.0, tk.END)
        self.txt_log.config(state='disabled')
        
        # Loga uma mensagem de confirmação
        self._log("--- Parâmetros resetados para o padrão ---")

   

    def _run(self):
        # 1. Tentamos ler TODOS os parâmetros de uma vez
        try:
            # Gera as seeds automaticamente
            base_seed = int(self.seeds_var.get()) 
            n_rep = int(self.n_reps_var.get())    
            seeds = [base_seed + i for i in range(n_rep)]
            
            # Lê os parâmetros básicos
            n_islands = int(self.n_islands_var.get())
            mig_rate = float(self.mig_rate_var.get())
            max_gen = int(self.max_gen_var.get())
            start_gen = int(self.start_gen_var.get())
            n_cores = int(self.n_cores_var.get())
            out_dir = self.out_dir_var.get()
            
            # Lê os parâmetros avançados (ADICIONEI OS QUE FALTAVAM AQUI)
            mut_rate = float(self.mut_rate_var.get())
            dbscan_eps = float(self.dbscan_eps_var.get())
            dbscan_min = int(self.dbscan_min_var.get())
            alpha = float(self.alpha_var.get())
            loci = int(self.loci_var.get())
            states = int(self.states_var.get())
            females = int(self.females_var.get())
            males = int(self.males_var.get())
            femsort = int(self.femsort_var.get())
            preselec = int(self.preselec_var.get())

        except ValueError as e:
            messagebox.showerror("Erro de Validação", f"Verifique os valores inseridos. Erro: {e}")
            return

        # 2. Preparamos a interface
        self.btn_run.config(state='disabled')
        self._log(f"--- INICIANDO SIMULAÇÃO ---")
        self._log(f"Base Seed: {base_seed} | Réplicas: {n_rep} (Sementes: {seeds[0]} a {seeds[-1]})")
        self._log(f"Ilhas: {n_islands} | Migração: {mig_rate} | Gerações: {max_gen}")
        self._log(f"DBSCAN eps: {dbscan_eps} | min_samples: {dbscan_min}")
        
        # 3. Criamos a função worker (em segundo plano para não travar a tela)
        def worker():
            try:
                # CRIAMOS O DICIONÁRIO OVERRIDES PARA ENVIAR TUDO PARA A SIMULAÇÃO
                overrides = {
                    'mutation_rate': mut_rate,
                    'dbscan_eps': dbscan_eps,
                    'dbscan_min': dbscan_min,
                    'n_loci': loci,
                    'n_states': states,
                    'n_females': females,
                    'n_males': males,
                    'FemSort': femsort,
                    'PreSelec': preselec
                }

                # Chamamos o run_all passando o overrides
                run_all(
                    out_dir=out_dir,
                    seeds=seeds,
                    n_islands=n_islands,
                    migration_rate=mig_rate,
                    max_gen=max_gen,
                    start_gen=start_gen,
                    n_cores=n_cores,
                    overrides=overrides, # <--- AQUI ESTÁ A MÁGICA
                    log_func=lambda m: self.after(0, self._log, m)
                )
                
                # Se terminar sem erros:
                self.after(0, lambda: messagebox.showinfo("Concluído", "Simulação finalizada com sucesso!"))
                self.after(0, self._log, "--- SIMULAÇÃO CONCLUÍDA COM SUCESSO ---")
                
            except Exception as e:
                # Se der erro:
                self.after(0, self._log, f"ERRO CRÍTICO: {e}")
                traceback.print_exc()
            finally:
                # Religa o botão no final
                self.after(0, lambda: self.btn_run.config(state='normal'))

        # 4. Inicia a thread
        threading.Thread(target=worker, daemon=True).start()



if __name__ == '__main__':
    app = IslandApp()
    app.mainloop()