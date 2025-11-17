# -- coding: utf-8 --
"""
SCRIPT FINAL: MARCHA INTELIGENTE (IMU) + sEMG COMPLETO + DIAGNÓSTICO
Versão com TODAS as métricas musculares (RMS, MAV, LOG, WL, MNF, MDF)
"""

import socket
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, iirnotch, butter, filtfilt
from scipy.fft import fft, fftfreq

# --- CONFIGURAÇÕES ---
HOST_IP = '0.0.0.0' 
HOST_PORT = 4210    
TEMPO_COLETA_SEGUNDOS = 15
COLUNAS_ESPERADAS = 7 

# --- CONFIGURAÇÕES sEMG ---
REGRAS_SEMG = {
    'aplicar_filtro_notch': True,
    'notch_freq': 60.0,
    'quality_factor': 30.0,
    'low_cut': 30.0,
    'high_cut': 450.0,
    'order': 4
}

# LISTA COMPLETA RESTAURADA
NOMES_PARAMETROS_SEMG = {
    'RMS': 'Força (RMS)',
    'MAV': 'Ativação (MAV)',
    'LOG': 'Índice (LOG)',
    'WL':  'Comp. Onda (WL)',
    'MNF': 'Freq. Média (MNF)',
    'MDF': 'Freq. Mediana (MDF)'
}

# =============================================================================
# --- FUNÇÕES AUXILIARES ---
# =============================================================================

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0: high = 0.99 
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq, Q, fs):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

# =============================================================================
# --- 1. PROCESSAMENTO DE MARCHA (IMU) ---
# =============================================================================

def processar_dados_marcha(dados_brutos, fs):
    print("\n[MARCHA] Iniciando análise cinemática...")
    
    df = pd.DataFrame(dados_brutos[:, :6], columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
    tempo = np.arange(len(df)) / fs
    
    # 1. Detecção Automática do Eixo
    std_g = df[['Gx', 'Gy', 'Gz']].std()
    eixo_dominante = std_g.idxmax()
    print(f"[MARCHA] Eixo de movimento detectado: {eixo_dominante}")
    
    sinal_raw = df[eixo_dominante].values
    sinal_filt = butter_lowpass(sinal_raw, 3, fs)
    
    # 2. Correção de Inversão
    fator_inversao = 1
    if np.abs(np.min(sinal_filt)) > np.max(sinal_filt):
        sinal_processamento = -sinal_filt
        fator_inversao = -1
    else:
        sinal_processamento = sinal_filt
        
    # 3. Detecção de Picos
    altura_min = np.max(sinal_processamento) * 0.25 
    picos, _ = find_peaks(sinal_processamento, height=altura_min, distance=fs*0.5)
    
    # 4. Busca de Eventos
    hs_times = []
    to_times = []
    hs_ind = []
    to_ind = [] 
    janela = int(0.4 * fs)

    for p in picos:
        inicio = max(0, p - janela)
        segmento_pre = sinal_processamento[inicio:p]
        if len(segmento_pre) > 0:
            idx_real = inicio + np.argmin(segmento_pre)
            to_times.append(tempo[idx_real])
            to_ind.append(idx_real)

        fim = min(len(sinal_processamento), p + janela)
        segmento_pos = sinal_processamento[p:fim]
        if len(segmento_pos) > 0:
            idx_real = p + np.argmin(segmento_pos)
            hs_times.append(tempo[idx_real])
            hs_ind.append(idx_real)
            
    # 5. Métricas
    media_apoio = np.nan
    media_balanco = np.nan
    cadencia = np.nan
    pct_apoio = np.nan
    estabilidade = np.nan
    
    if len(hs_times) > 1:
        stride_durations = np.diff(hs_times)
        media_stride = np.mean(stride_durations)
        
        min_len = min(len(hs_times), len(to_times))
        swing_durations = np.array(hs_times[:min_len]) - np.array(to_times[:min_len])
        swing_durations = swing_durations[swing_durations > 0]
        
        if len(swing_durations) > 0 and media_stride > 0:
            media_balanco = np.mean(swing_durations)
            media_apoio = media_stride - media_balanco
            cadencia = 120.0 / media_stride
            pct_apoio = (media_apoio / media_stride) * 100.0
            estabilidade = (np.std(stride_durations) / media_stride) * 100

    metricas = {
        'media_apoio': media_apoio,
        'media_balanco': media_balanco,
        'cadencia': cadencia,
        'pct_apoio': pct_apoio,
        'estabilidade': estabilidade,
        'tempo': tempo,
        'sinal_plot': sinal_filt, 
        'nome_eixo': eixo_dominante,
        'hs_ind': np.array(hs_ind),
        'to_ind': np.array(to_ind),
        'picos_ind': picos
    }
    return metricas

# =============================================================================
# --- 2. PROCESSAMENTO sEMG (COMPLETO) ---
# =============================================================================

def calculate_emg_parameters(data, fs):
    params = {}
    data_safe = np.abs(data) + 1e-10
    
    # Domínio do Tempo (Métricas Básicas + WL + LOG)
    params['RMS'] = np.sqrt(np.mean(data**2))
    params['MAV'] = np.mean(np.abs(data))
    params['LOG'] = np.exp(np.mean(np.log(data_safe)))
    params['WL']  = np.sum(np.abs(np.diff(data)))
    
    # Domínio da Frequência (MNF + MDF)
    N = len(data)
    yf = fft(data)
    psd = (1/(N*fs)) * np.abs(yf[0:N//2])**2
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    sum_psd = np.sum(psd)
    if sum_psd > 0:
        params['MNF'] = np.sum(xf * psd) / sum_psd
        cumulative_power = np.cumsum(psd)
        median_freq_index = np.where(cumulative_power >= sum_psd / 2)[0]
        params['MDF'] = xf[median_freq_index[0]] if len(median_freq_index) > 0 else 0
    else:
        params['MNF'] = 0
        params['MDF'] = 0
        
    return params

def processar_dados_semg(sinal_bruto, fs, configs):
    print(f"\n[sEMG] Processando sinal muscular...")
    dados = sinal_bruto
    if configs['aplicar_filtro_notch']:
        dados = notch_filter(dados, configs['notch_freq'], configs['quality_factor'], fs)
    dados_filtrados = butter_bandpass_filter(dados, configs['low_cut'], configs['high_cut'], fs, configs['order'])
    
    mx = np.max(np.abs(dados_filtrados))
    dados_norm = dados_filtrados / mx if mx > 0 else dados_filtrados
    
    params = calculate_emg_parameters(dados_norm, fs)
    return params, dados_filtrados

# =============================================================================
# --- 3. RELATÓRIO FINAL ---
# =============================================================================

def gerar_relatorio(metricas, params_semg, sinal_semg, fs, nome_arq):
    print(f"\nGerando relatório completo: {nome_arq}")

    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('#f0f0f0')
    fig.suptitle("Análise Biomecânica: Marcha & Eletromiografia", fontsize=26, weight='bold')

    ax_txt_imu = axs[0, 0]
    ax_txt_semg = axs[0, 1]
    ax_plot_imu = axs[1, 0]
    ax_plot_semg = axs[1, 1]

    # DIAGNÓSTICO AUTOMÁTICO
    cad = metricas['cadencia']
    apoio_pct = metricas['pct_apoio']
    diagnostico = "Indefinido"
    cor_diag = "gray"
    
    if not np.isnan(cad):
        if cad > 85 and apoio_pct < 66:
            diagnostico = "SAUDÁVEL / NORMAL"
            cor_diag = "green"
        elif cad < 60:
            diagnostico = "ATÍPICA (Lenta)"
            cor_diag = "red"
        elif apoio_pct > 70:
            diagnostico = "ATÍPICA (Instável)"
            cor_diag = "red"
        else:
            diagnostico = "LEVEMENTE ANORMAL"
            cor_diag = "#e67e00"

    # BLOCO 1: TEXTO IMU
    ax_txt_imu.set_title("1. Relatório da Marcha", fontsize=18, weight='bold')
    ax_txt_imu.axis('off')
    y = 0.85
    def ptxt(ax, l, v, c="black", w='normal', sz=14):
        nonlocal y
        ax.text(0.05, y, l, fontsize=sz, weight='bold')
        ax.text(0.6, y, v, fontsize=sz, color=c, weight=w)
        y -= 0.12

    ptxt(ax_txt_imu, "Cadência:", f"{cad:.1f} ppm" if not np.isnan(cad) else "N/A")
    ptxt(ax_txt_imu, "Tempo de Balanço:", f"{metricas['media_balanco']:.3f} s" if not np.isnan(metricas['media_balanco']) else "N/A")
    ptxt(ax_txt_imu, "Tempo de Apoio:", f"{metricas['media_apoio']:.3f} s" if not np.isnan(metricas['media_apoio']) else "N/A")
    ptxt(ax_txt_imu, "% do Ciclo em Apoio:", f"{apoio_pct:.1f} %" if not np.isnan(apoio_pct) else "N/A")

    y -= 0.05
    ax_txt_imu.text(0.05, y, "DIAGNÓSTICO:", fontsize=14, weight='bold')
    ax_txt_imu.text(0.05, y-0.1, diagnostico, fontsize=20, weight='bold', color=cor_diag, bbox=dict(facecolor='white', alpha=0.9, edgecolor=cor_diag))

    # BLOCO 2: TEXTO sEMG (AGORA COMPLETO)
    ax_txt_semg.set_title("2. Métricas Musculares (Todas)", fontsize=18, weight='bold')
    ax_txt_semg.axis('off')
    y = 0.90 # Começa mais alto para caber 6 linhas
    
    # Itera sobre a lista completa
    for k, v_nome in NOMES_PARAMETROS_SEMG.items():
        valor = params_semg.get(k, 0.0)
        ax_txt_semg.text(0.05, y, f"{v_nome}:", fontsize=13, weight='bold')
        ax_txt_semg.text(0.6, y, f"{valor:.4f}", fontsize=13)
        y -= 0.10 # Espaçamento um pouco menor para caber tudo

    # BLOCO 3: PLOT IMU
    eixo = metricas['nome_eixo']
    ax_plot_imu.set_title(f"Cinemática ({eixo}) - Fases Detectadas", fontsize=14, weight='bold')
    t = metricas['tempo']
    s = metricas['sinal_plot']
    ax_plot_imu.plot(t, s, 'k-', alpha=0.6, label='Giroscópio')
    
    pk = metricas['picos_ind']
    hs = metricas['hs_ind']
    to = metricas['to_ind']
    
    if len(pk)>0: ax_plot_imu.plot(t[pk], s[pk], 'rx', label='Mid-Swing')
    if len(hs)>0: ax_plot_imu.plot(t[hs], s[hs], 'bo', label='Heel Strike')
    if len(to)>0: ax_plot_imu.plot(t[to], s[to], 'go', label='Toe Off')
    
    ax_plot_imu.set_xlabel('Tempo (s)')
    ax_plot_imu.legend(loc='upper right')
    ax_plot_imu.grid(True, linestyle='--')

    # BLOCO 4: PLOT sEMG
    ax_plot_semg.set_title("Ativação Muscular (Filtrado)", fontsize=14, weight='bold')
    ts = np.arange(len(sinal_semg))/fs
    ax_plot_semg.plot(ts, sinal_semg, color='#ff7f0e', lw=0.8)
    ax_plot_semg.set_xlabel('Tempo (s)')
    ax_plot_semg.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(nome_arq, dpi=150)
    plt.close(fig)

# =============================================================================
# --- MAIN ---
# =============================================================================

def main():
    dados = []
    print(f"--- SERVIDOR UDP PRONTO {HOST_IP}:{HOST_PORT} ---")
    print(f"Caminhe por {TEMPO_COLETA_SEGUNDOS} segundos...")
    
    start_t = 0
    started = False
    
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((HOST_IP, HOST_PORT))
    udp.settimeout(3.0)
    
    try:
        while True:
            try:
                data, addr = udp.recvfrom(1024)
                if not started:
                    print(f"Conectado! Coletando...")
                    start_t = time.time()
                    started = True
                
                if started and (time.time() - start_t > TEMPO_COLETA_SEGUNDOS):
                    break
                    
                line = data.decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == COLUNAS_ESPERADAS:
                    dados.append([float(x) for x in parts])
            except socket.timeout:
                if started: break
    finally:
        udp.close()
        
    if len(dados) > 50:
        arr = np.array(dados)
        fs = len(dados) / (time.time() - start_t)
        print(f"Freq. Amostragem: {fs:.1f} Hz")
        
        metrics = processar_dados_marcha(arr, fs)
        p_emg, s_emg = processar_dados_semg(arr[:, 6], fs, REGRAS_SEMG)
        
        arquivo = f"analise_completa_{int(time.time())}.png"
        gerar_relatorio(metrics, p_emg, s_emg, fs, arquivo)
        print(f"RELATÓRIO SALVO: {arquivo}")
    else:
        print("Dados insuficientes.")

if _name_ == "_main_":
    main()
