import numpy as np
import matplotlib.pyplot as plt

# Parametri vozil
kona_1_6T = {
    "ime": "2025 Kona 1.6T 2WD (170 KM)",
    "masa_kg": 1450,
    "koef_upora": 0.27,
    "frontalna_povrsina": 2.49,
    "polmer_kolesa": 0.344,
    "ucinkovitost_pogona": 0.92,
    "prestave": [4.717, 2.906, 1.864, 1.423, 1.224, 1.000, 0.790, 0.635],
    "koncni_prenos": 3.510,
    "cas_prestavljanja": 0.4,
    "max_vrtljaji": 6200,
    "barva": "blue"
}

kona_1_0T = {
    "ime": "2024 Kona 1.0T FWD (120 KM)",
    "masa_kg": 1410,
    "koef_upora": 0.27,
    "frontalna_povrsina": 2.45,
    "polmer_kolesa": 0.343,
    "ucinkovitost_pogona": 0.90,
    "prestave": [3.643, 2.174, 1.826, 1.024, 0.809, 0.854, 0.717],
    "koncni_prenos": {1: 4.643, 2: 4.643, 3: 3.611, 4: 4.643, 5: 4.643, 6: 3.611, 7: 3.611},
    "cas_prestavljanja": 0.25,
    "max_vrtljaji": 6200,
    "barva": "green"
}

gostota_zraka = 1.225
gravitacija = 9.81
koef_kotaljenja = 0.012

def moment_1_6T(vrtljaji):
    if vrtljaji < 1900:
        return np.interp(vrtljaji, [1000, 1900], [105, 218])
    elif vrtljaji <= 4000:
        return 218
    elif vrtljaji <= 6500:
        return np.interp(vrtljaji, [4000, 6000, 6500], [218, 199, 168])
    else:
        return 0

def moment_1_0T(vrtljaji):
    if vrtljaji < 2000:
        return np.interp(vrtljaji, [1000, 2000], [100, 200])
    elif vrtljaji <= 3500:
        return 200
    elif vrtljaji <= 6000:
        return np.interp(vrtljaji, [3500, 6000], [200, 120])
    else:
        return 80

def izracunaj_moc(moment_nm, vrtljaji):
    if vrtljaji == 0:
        return 0
    return (moment_nm * vrtljaji * 2 * np.pi / 60) / 1000

def narisi_krivulje_motorja():
    obseg_vrtljajev = np.linspace(1000, 6500, 500)
    
    moment_16 = [moment_1_6T(rpm) for rpm in obseg_vrtljajev]
    moc_16 = [izracunaj_moc(moment, rpm) for moment, rpm in zip(moment_16, obseg_vrtljajev)]
    
    moment_10 = [moment_1_0T(rpm) for rpm in obseg_vrtljajev]
    moc_10 = [izracunaj_moc(moment, rpm) for moment, rpm in zip(moment_10, obseg_vrtljajev)]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Vrtljaji motorja (min⁻¹)')
    ax1.set_ylabel('Navor (Nm)', color='tab:blue')
    ax1.plot(obseg_vrtljajev, moment_16, color=kona_1_6T["barva"], linestyle='-', label='1.6T Navor')
    ax1.plot(obseg_vrtljajev, moment_10, color=kona_1_0T["barva"], linestyle='-', label='1.0T Navor')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Moč (kW)', color='tab:red')
    ax2.plot(obseg_vrtljajev, moc_16, color=kona_1_6T["barva"], linestyle='--', label='1.6T Moč')
    ax2.plot(obseg_vrtljajev, moc_10, color=kona_1_0T["barva"], linestyle='--', label='1.0T Moč')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('Krivulje zmogljivosti motorja', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.85))
    plt.show()

def simulacija_pospeska(parametri, funkcija_momenta, ciljna_hitrost=160):
    masa = parametri["masa_kg"]
    cd = parametri["koef_upora"] 
    povrsina = parametri["frontalna_povrsina"]
    polmer = parametri["polmer_kolesa"]
    ucinkovitost = parametri["ucinkovitost_pogona"]
    prestave = parametri["prestave"]
    max_rpm = parametri["max_vrtljaji"]
    cas_menjave = parametri["cas_prestavljanja"]
    
    if isinstance(parametri["koncni_prenos"], dict):
        koncni_prenos = parametri["koncni_prenos"]
    else:
        koncni_prenos = {i+1: parametri["koncni_prenos"] for i in range(len(prestave))}
    
    korak_casa = 0.01
    hitrost_ms = 0.01
    cas = 0.0
    trenutna_prestava = 1
    
    casi, hitrosti, vrtljaji_podatki, prestave_podatki = [], [], [], []

    while hitrost_ms * 3.6 < ciljna_hitrost:
        skupni_prenos = prestave[trenutna_prestava - 1] * koncni_prenos[trenutna_prestava]
        vrtljaji = (hitrost_ms * 60) / (2 * np.pi * polmer) * skupni_prenos
        
        if vrtljaji >= max_rpm and trenutna_prestava < len(prestave):
            cas += cas_menjave
            trenutna_prestava += 1
            skupni_prenos = prestave[trenutna_prestava - 1] * koncni_prenos[trenutna_prestava]
            vrtljaji = (hitrost_ms * 60) / (2 * np.pi * polmer) * skupni_prenos

        moment_motorja = funkcija_momenta(vrtljaji)
        vlecna_sila = (moment_motorja * skupni_prenos * ucinkovitost) / polmer
        aero_upor = 0.5 * gostota_zraka * cd * povrsina * hitrost_ms**2
        kotaljni_upor = koef_kotaljenja * masa * gravitacija
        neto_sila = vlecna_sila - (aero_upor + kotaljni_upor)
        
        if neto_sila < 0: 
            neto_sila = 0

        pospesek = neto_sila / masa
        hitrost_ms += pospesek * korak_casa
        cas += korak_casa
        
        casi.append(cas)
        hitrosti.append(hitrost_ms * 3.6)
        vrtljaji_podatki.append(vrtljaji)
        prestave_podatki.append(trenutna_prestava)

    return casi, hitrosti, vrtljaji_podatki, prestave_podatki

rezultati_16 = simulacija_pospeska(kona_1_6T, moment_1_6T)
rezultati_10 = simulacija_pospeska(kona_1_0T, moment_1_0T)

def narisi_pospesek():
    plt.figure(figsize=(12, 7))
    plt.plot(rezultati_16[0], rezultati_16[1], label=kona_1_6T["ime"], color=kona_1_6T["barva"], linewidth=2)
    plt.plot(rezultati_10[0], rezultati_10[1], label=kona_1_0T["ime"], color=kona_1_0T["barva"], linewidth=2)
    
    cas_100_16 = np.interp(100, rezultati_16[1], rezultati_16[0])
    cas_100_10 = np.interp(100, rezultati_10[1], rezultati_10[0])
    cas_120_16 = np.interp(120, rezultati_16[1], rezultati_16[0])
    cas_120_10 = np.interp(120, rezultati_10[1], rezultati_10[0])
    cas_130_16 = np.interp(130, rezultati_16[1], rezultati_16[0])
    cas_130_10 = np.interp(130, rezultati_10[1], rezultati_10[0])
    cas_140_16 = np.interp(140, rezultati_16[1], rezultati_16[0])
    cas_140_10 = np.interp(140, rezultati_10[1], rezultati_10[0])
    
    plt.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    plt.text(cas_100_16 + 0.2, 98, f'{cas_100_16:.1f}s', va='top', ha='left', color=kona_1_6T["barva"], weight='bold')
    plt.text(cas_100_10 + 0.2, 98, f'{cas_100_10:.1f}s', va='top', ha='left', color=kona_1_0T["barva"], weight='bold')
    
    plt.axhline(120, color='gray', linestyle='--', linewidth=0.8)
    plt.text(cas_120_16 + 0.2, 118, f'{cas_120_16:.1f}s', va='top', ha='left', color=kona_1_6T["barva"], weight='bold')
    plt.text(cas_120_10 + 0.2, 118, f'{cas_120_10:.1f}s', va='top', ha='left', color=kona_1_0T["barva"], weight='bold')
    
    plt.axhline(130, color='gray', linestyle='--', linewidth=0.8)
    plt.text(cas_130_16 + 0.2, 128, f'{cas_130_16:.1f}s', va='top', ha='left', color=kona_1_6T["barva"], weight='bold')
    plt.text(cas_130_10 + 0.2, 128, f'{cas_130_10:.1f}s', va='top', ha='left', color=kona_1_0T["barva"], weight='bold')
    
    plt.axhline(140, color='gray', linestyle='--', linewidth=0.8)
    plt.text(cas_140_16 + 0.2, 138, f'{cas_140_16:.1f}s', va='top', ha='left', color=kona_1_6T["barva"], weight='bold')
    plt.text(cas_140_10 + 0.2, 138, f'{cas_140_10:.1f}s', va='top', ha='left', color=kona_1_0T["barva"], weight='bold')
    
    plt.title('Simulacija pospeška (0-160 km/h)', fontsize=16)
    plt.xlabel('Čas (sekunde)')
    plt.ylabel('Hitrost (km/h)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xlim(0, max(rezultati_10[0]))
    plt.ylim(0, 165)
    plt.show()

def narisi_vlecno_silo():
    obseg_hitrosti = np.linspace(1, 150, 200)
    hitrost_ms = obseg_hitrosti / 3.6
    povprecna_masa = (kona_1_6T["masa_kg"] + kona_1_0T["masa_kg"]) / 2
    sila_upora = (0.5 * gostota_zraka * kona_1_6T["koef_upora"] * kona_1_6T["frontalna_povrsina"] * hitrost_ms**2) + (koef_kotaljenja * povprecna_masa * gravitacija)

    plt.figure(figsize=(14, 8))
    plt.plot(obseg_hitrosti, sila_upora, label='Uporne sile (aero + kotaljenje)', color='red', linewidth=2.5, zorder=10)

    for prestava in range(1, 6):
        skupni_prenos_16 = kona_1_6T["prestave"][prestava-1] * kona_1_6T["koncni_prenos"]
        vrtljaji_16 = (hitrost_ms * 60) / (2 * np.pi * kona_1_6T["polmer_kolesa"]) * skupni_prenos_16
        moment_16 = np.array([moment_1_6T(rpm) for rpm in vrtljaji_16])
        vlecna_sila_16 = (moment_16 * skupni_prenos_16 * kona_1_6T["ucinkovitost_pogona"]) / kona_1_6T["polmer_kolesa"]
        veljaven_obseg_16 = (vrtljaji_16 > 1000) & (vrtljaji_16 <= kona_1_6T["max_vrtljaji"])
        plt.plot(obseg_hitrosti[veljaven_obseg_16], vlecna_sila_16[veljaven_obseg_16], color=kona_1_6T["barva"], linestyle='-', label=f'1.6T' if prestava==1 else "")

        koncni_prenos_10 = kona_1_0T["koncni_prenos"][prestava]
        skupni_prenos_10 = kona_1_0T["prestave"][prestava-1] * koncni_prenos_10
        vrtljaji_10 = (hitrost_ms * 60) / (2 * np.pi * kona_1_0T["polmer_kolesa"]) * skupni_prenos_10
        moment_10 = np.array([moment_1_0T(rpm) for rpm in vrtljaji_10])
        vlecna_sila_10 = (moment_10 * skupni_prenos_10 * kona_1_0T["ucinkovitost_pogona"]) / kona_1_0T["polmer_kolesa"]
        veljaven_obseg_10 = (vrtljaji_10 > 1000) & (vrtljaji_10 <= kona_1_0T["max_vrtljaji"])
        plt.plot(obseg_hitrosti[veljaven_obseg_10], vlecna_sila_10[veljaven_obseg_10], color=kona_1_0T["barva"], linestyle='--', label=f'1.0T' if prestava==1 else "")

    plt.title('Vlečna sila v odvisnosti od hitrosti vozila', fontsize=16)
    plt.xlabel('Hitrost (km/h)')
    plt.ylabel('Sila (Newton)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.ylim(0, 18000)
    plt.xlim(0, 150)
    plt.show()

def narisi_vrtljaje_pri_vožnji():
    hitrost_kmh = np.linspace(60, 130, 100)
    hitrost_ms = hitrost_kmh / 3.6

    skupni_prenos_16 = kona_1_6T["prestave"][-1] * kona_1_6T["koncni_prenos"]
    vrtljaji_16 = (hitrost_ms * 60) / (2 * np.pi * kona_1_6T["polmer_kolesa"]) * skupni_prenos_16

    skupni_prenos_10 = kona_1_0T["prestave"][-1] * kona_1_0T["koncni_prenos"][7]
    vrtljaji_10 = (hitrost_ms * 60) / (2 * np.pi * kona_1_0T["polmer_kolesa"]) * skupni_prenos_10

    plt.figure(figsize=(12, 7))
    plt.plot(hitrost_kmh, vrtljaji_16, label=f'{kona_1_6T["ime"]} (8. prestava)', color=kona_1_6T["barva"])
    plt.plot(hitrost_kmh, vrtljaji_10, label=f'{kona_1_0T["ime"]} (7. prestava)', color=kona_1_0T["barva"])
    
    vrtljaji_120_16 = np.interp(120, hitrost_kmh, vrtljaji_16)
    vrtljaji_120_10 = np.interp(120, hitrost_kmh, vrtljaji_10)
    plt.axvline(120, color='red', linestyle=':', linewidth=1)
    plt.scatter([120, 120], [vrtljaji_120_16, vrtljaji_120_10], color='red', zorder=5)
    plt.text(120.5, vrtljaji_120_16, f' {vrtljaji_120_16:.0f} min⁻¹', ha='left', va='center', color=kona_1_6T["barva"])
    plt.text(120.5, vrtljaji_120_10, f' {vrtljaji_120_10:.0f} min⁻¹', ha='left', va='center', color=kona_1_0T["barva"])

    plt.title('Vrtljaji motorja pri vožnji po avtocesti (najvišja prestava)', fontsize=16)
    plt.xlabel('Hitrost (km/h)')
    plt.ylabel('Vrtljaji motorja (min⁻¹)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    print("Generiranje grafov zmogljivosti za Hyundai Kona 1.6T 2WD vs 1.0T FWD...")
    narisi_krivulje_motorja()
    narisi_pospesek()
    narisi_vlecno_silo()
    narisi_vrtljaje_pri_vožnji()
    print("končano.")
