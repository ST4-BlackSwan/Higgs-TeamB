# Données issues de la partie Systematics

#S_change = {"tes": [(\Delta_S_tes_1(0.97),\Delta_S_tes_1(1.03)), (\Delta_S_tes_2(0.97),\Delta_S_tes_2(1.03)),(\Delta_S_tes_3(0.97),\Delta_S_tes_3(1.03))],
#       "jes": [(\Delta_JES_1(0.97),\Delta_JES_1(1.03)), (\Delta_JES_2(0.97),\Delta_JES_2(1.03)),(\Delta_JES_3(0.97),\Delta_JES_3(1.03))],
#       "bnorm": [(\Delta_BNORM_1(0.95),\Delta_BNORM_1(1.05)), (\Delta_BNORM_2(0.95),\Delta_BNORM_2(1.05)),(\Delta_BNORM_3(0.97),\Delta_BNORM_3(1.03))],
#       "smet" : [(\Delta_SMET_1(-3),\Delta_SMET_1(3)), (\Delta_SMET_2(-3),\Delta_SMET_2(3)),(\Delta_SMET_3(-3),\Delta_SMET_3(3))],}

#B_change = {"tes": [(\Delta_B_tes_1(0.97),\Delta_B_tes_1(1.03)), (\Delta_B_tes_2(0.97),\Delta_B_tes_2(1.03)),(\Delta_B_tes_3(0.97),\Delta_B_tes_3(1.03))],
#       "jes": [(\Delta_B_JES_1(0.97),\Delta_B_JES_1(1.03)), (\Delta_B_JES_2(0.97),\Delta_B_JES_2(1.03)),(\Delta_B_JES_3(0.97),\Delta_B_JES_3(1.03))],
#       "bnorm": [(\Delta_B_BNORM_1(0.95),\Delta_B_BNORM_1(1.05)), (\Delta_B_BNORM_2(0.95),\Delta_B_BNORM_2(1.05)),(\Delta_B_BNORM_3(0.95),\Delta_B_BNORM_3(1.05))],
#       "smet" : [(\Delta_B_SMET_1(-3),\Delta_B_SMET_1(3)), (\Delta_B_SMET_2(-3),\Delta_B_SMET_2(3)),(\Delta_B_SMET_3(-3),\Delta_B_SMET_3(3))],}

evaluation={"tes":(0.97,1.03),
            "jes":(0.97,1.03),
            "bnorm":(0.95,1.05),
            "smet":(-3,3)}

S_change={"tes":[(0.19699097,0.70884323),(-1.7246513,0.9830055),(-17.686646,16.813202)], "jes":[(0.5398369,-0.35663986),(0.2844162,-0.26097488),(-7.834198,7.358368)], "bnorm":[(),(),()], "smet":[(0,-0.058582306),(0,0.04607773),(0,0.67333984)]}
B_change={"tes":[(-270.76428,308.10718),(-289.75366,372.21924),(-492.30054,598.72754)], "jes":[(-89.5376,67.917725),(-98.22339,125.55737),(-226.47021,255.14648)], "bnorm":[], "smet":[(0,85.43237),(0,68.12305),(0,87.16016)]}

def get_S (bin, systematic,value):
    return  (S_change[systematic][bin][1]-S_change[systematic][bin][0])/(evaluation[systematic][1]-evaluation[systematic][0])*(value-evaluation[systematic][0])+S_change[systematic][bin][0]
    
    
def get_B (bin, systematic,value):
    return  (B_change[systematic][bin][1]-B_change[systematic][bin][0])/(evaluation[systematic][1]-evaluation[systematic][0])*(value-evaluation[systematic][0])+B_change[systematic][bin][0]
    