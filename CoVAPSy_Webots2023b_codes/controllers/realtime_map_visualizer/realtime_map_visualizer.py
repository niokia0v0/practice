# -*- coding: utf-8 -*-

"""
visualisateur_carte_temps_reel.py - Contrôleur SLAM pour véhicule Webots Covapsy
"""

from vehicle import Driver
from controller import Lidar, Gyro
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import time
from scipy.spatial import KDTree

# --- Initialisation ---
driver = Driver()
timestep = int(driver.getBasicTimeStep())

lidar = Lidar("RpLidarA2")
lidar.enable(timestep)
lidar.enablePointCloud()

gyro = Gyro("gyro")
gyro.enable(timestep)

# Constantes du véhicule
VITESSE_AUTOPILOTE_M_S = 0.41  # Vitesse originale du code de référence
ANGLE_DIRECTION_MAX_RAD = 0.3185
LARGEUR_VEHICULE = 0.38  # 38cm
RAYON_FILTRE = 0.19  # 19cm
DISTANCE_DEPLACEMENT = 0.5  # 0.5m entre chaque scan

# Paramètres de carte
RESOLUTION_CARTE = 0.01  # m par pixel

# Paramètres de calibration et confiance
RAYON_CORRESPONDANCE = 0.05  # 5cm pour considérer qu'un point correspond
DUREE_ATTENTE_AVANT_SCAN = 1.5  # 1.5s d'attente après arrêt complet
DUREE_SCAN_COMPARAISON = 1.0  # Scanner pendant 1s pour comparer
SEUIL_VITESSE_ARRET = 0.01  # m/s et rad/s pour considérer l'arrêt

# --- État de l'odométrie ---
pose_robot = np.array([0.0, 0.0, 0.0])  # x, y, angle
temps_precedent = 0.0
est_calibre = False
biais_gyro = 0.0
echantillons_calibration = []
DUREE_CALIBRATION = 3.0

# --- État du système ---
est_en_pause = False
programme_termine = False
derniere_mise_a_jour_affichage = 0.0
INTERVALLE_AFFICHAGE = 0.2  # 0.2s pour l'affichage des points bleus

# --- État stop-and-scan ---
mode_scan = "CALIBRATION_GYRO"
position_debut_deplacement = None
distance_parcourue = 0.0
temps_arret_complet = None
temps_debut_comparaison = None
scans_comparaison = []
kdtree_carte = None
premier_scan_effectue = False

# Stockage des données
points_carte_globale = []

# Variables pour le suivi de la vitesse
vitesse_precedente = 0.0
vitesse_angulaire_precedente = 0.0

# ==================== Configuration Matplotlib ====================
plt.ion()
fig = plt.figure(figsize=(12, 10))
ax_carte = fig.add_subplot(111)
ax_carte.set_aspect('equal')
ax_carte.grid(True, alpha=0.3)
ax_carte.set_title('Carte SLAM en Temps Réel')
ax_carte.set_xlabel('X (m)')
ax_carte.set_ylabel('Y (m)')

# Créer les boutons
fig.subplots_adjust(bottom=0.15)

# Bouton pause
ax_pause = plt.axes([0.7, 0.02, 0.1, 0.04])
btn_pause = Button(ax_pause, 'Pause')

# Bouton sauvegarde
ax_sauvegarde = plt.axes([0.81, 0.02, 0.1, 0.04])
btn_sauvegarde = Button(ax_sauvegarde, 'Sauvegarder')

# =====================================================================

def basculer_pause(event):
    """Bascule l'état pause/reprise"""
    global est_en_pause
    est_en_pause = not est_en_pause
    btn_pause.label.set_text('Reprendre' if est_en_pause else 'Pause')
    
    if est_en_pause:
        driver.setCruisingSpeed(0)
        print("Programme en pause")
    else:
        print("Reprise du programme")

def sauvegarder_carte_et_fermer(event):
    """Sauvegarde la carte en format BMP et ferme l'application"""
    global points_carte_globale, programme_termine
    
    if not points_carte_globale:
        print("Aucune donnée de carte à sauvegarder")
        return
    
    print("Sauvegarde de la carte en cours...")
    
    # Arrêter le véhicule
    driver.setCruisingSpeed(0)
    
    # Convertir les points en tableau numpy
    points_np = np.array(points_carte_globale)
    
    # Trouver les limites de la carte
    x_min, y_min = points_np.min(axis=0)
    x_max, y_max = points_np.max(axis=0)
    
    # Ajouter une marge
    marge = 0.5  # 50cm de marge
    x_min -= marge
    y_min -= marge
    x_max += marge
    y_max += marge
    
    # Calculer la taille de l'image en pixels
    largeur_pixels = int((x_max - x_min) / RESOLUTION_CARTE)
    hauteur_pixels = int((y_max - y_min) / RESOLUTION_CARTE)
    
    print(f"Taille de l'image: {largeur_pixels}x{hauteur_pixels} pixels")
    print(f"Zone couverte: {x_max-x_min:.2f}m x {y_max-y_min:.2f}m")
    
    # Créer l'image binaire
    image_carte = np.ones((hauteur_pixels, largeur_pixels), dtype=np.uint8) * 255
    
    # Convertir les points en coordonnées pixel
    for point in points_carte_globale:
        px = int((point[0] - x_min) / RESOLUTION_CARTE)
        py = int((point[1] - y_min) / RESOLUTION_CARTE)
        
        # Inverser l'axe Y pour l'image
        py = hauteur_pixels - 1 - py
        
        # Vérifier les limites et marquer le pixel
        if 0 <= px < largeur_pixels and 0 <= py < hauteur_pixels:
            """
            # Marquer un petit carré pour rendre les obstacles plus visibles
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < largeur_pixels and 0 <= ny < hauteur_pixels:
                        image_carte[ny, nx] = 0  # Noir = obstacle
            """
            # Ou simplement marquer le pixel
            image_carte[py, px] = 0  # Noir = obstacle    
    
    # Sauvegarder l'image
    img = Image.fromarray(image_carte)
    nom_fichier = f"carte_slam_{time.strftime('%Y%m%d_%H%M%S')}.bmp"
    img.save(nom_fichier)
    print(f"Carte sauvegardée: {nom_fichier}")
    
    # Marquer pour terminer le programme
    programme_termine = True
    
    # Fermer matplotlib
    plt.close('all')

# Connecter les boutons
btn_pause.on_clicked(basculer_pause)
btn_sauvegarde.on_clicked(sauvegarder_carte_et_fermer)

def nettoyer_donnees_lidar():
    """Nettoie les données lidar en remplaçant les valeurs invalides"""
    distances = lidar.getRangeImage()
    distances_np = np.array(distances)
    distances_np[np.isinf(distances_np) | np.isnan(distances_np)] = lidar.getMaxRange()
    return distances_np.tolist()

def calculer_angle_direction(donnees_lidar):
    """Calcule l'angle de direction basé sur les mesures lidar (angles 120-240)"""
    angle_base = (donnees_lidar[240] - donnees_lidar[120]) * 1.5
    return max(-ANGLE_DIRECTION_MAX_RAD, min(ANGLE_DIRECTION_MAX_RAD, angle_base))

def filtrer_points_lidar(nuage_points):
    """Filtre les points dans le rayon du véhicule (19cm)"""
    points_filtres = []
    for point in nuage_points:
        if point is None or math.isinf(point.x) or math.isnan(point.x):
            continue
        
        distance = math.sqrt(point.x**2 + point.y**2)
        
        if distance > RAYON_FILTRE:
            points_filtres.append(point)
    
    return points_filtres

def obtenir_points_lidar_temps_reel():
    """Obtient les points lidar actuels pour l'affichage en temps réel"""
    x_robot, y_robot, angle_robot = pose_robot[0], pose_robot[1], pose_robot[2]
    
    nuage_points_lidar = lidar.getPointCloud()
    points_filtres = filtrer_points_lidar(nuage_points_lidar)
    
    cos_angle = math.cos(angle_robot)
    sin_angle = math.sin(angle_robot)
    decalage_lidar_x = 0.2  # Position du lidar sur le véhicule
    
    points_globaux = []
    
    for point in points_filtres:
        vehicule_x = point.x + decalage_lidar_x
        vehicule_y = point.y
        
        global_x = x_robot + cos_angle * vehicule_x - sin_angle * vehicule_y
        global_y = y_robot + sin_angle * vehicule_x + cos_angle * vehicule_y
        
        points_globaux.append([global_x, global_y])
    
    return points_globaux

def calculer_score_confiance(points_scan):
    """Calcule le score de confiance d'un scan basé sur la correspondance avec la carte existante"""
    global kdtree_carte
    
    if not points_carte_globale or len(points_carte_globale) < 10:
        # Pas assez de points dans la carte pour comparer
        return 1.0
    
    # Créer ou mettre à jour le KDTree
    if kdtree_carte is None:
        kdtree_carte = KDTree(points_carte_globale)
    
    # Calculer le pourcentage de points qui correspondent
    nb_correspondances = 0
    for point in points_scan:
        distances, _ = kdtree_carte.query(point, k=1)
        if distances < RAYON_CORRESPONDANCE:
            nb_correspondances += 1
    
    # Score = ratio de points qui correspondent
    if len(points_scan) > 0:
        score = nb_correspondances / len(points_scan)
    else:
        score = 0.0
    
    return score

def effectuer_scan_initial():
    """Effectue le scan initial de référence"""
    global points_carte_globale, kdtree_carte, premier_scan_effectue
    
    print("Scan initial de référence...")
    scan_initial = obtenir_points_lidar_temps_reel()
    
    # Ajouter directement à la carte (sans test de confiance)
    points_carte_globale.extend(scan_initial)
    kdtree_carte = KDTree(points_carte_globale) if points_carte_globale else None
    premier_scan_effectue = True
    
    print(f"Scan initial effectué: {len(scan_initial)} points")
    return scan_initial

def est_vehicule_arrete(vitesse_actuelle, vitesse_angulaire_actuelle):
    """Vérifie si le véhicule est complètement arrêté"""
    return (abs(vitesse_actuelle) < SEUIL_VITESSE_ARRET and 
            abs(vitesse_angulaire_actuelle) < SEUIL_VITESSE_ARRET)

# Variables pour l'affichage
scan_actuel_affichage = []
points_temps_reel = []
compteur_images = 0

# --- Boucle principale ---
try:
    while driver.step() != -1 and not programme_termine:
        temps_actuel = driver.getTime()
        
        if temps_precedent == 0.0: 
            temps_precedent = temps_actuel
            continue
            
        dt = temps_actuel - temps_precedent
        temps_precedent = temps_actuel
        if dt == 0: 
            continue
        
        # Toujours mettre à jour l'odométrie
        if est_calibre:
            donnees_gyro = gyro.getValues()
            vitesse_actuelle = driver.getCurrentSpeed() / 3.6  # km/h vers m/s
            if math.isnan(vitesse_actuelle): 
                vitesse_actuelle = 0.0
            
            vitesse_angulaire_corrigee = donnees_gyro[2] - biais_gyro
            pose_robot[2] += vitesse_angulaire_corrigee * dt
            pose_robot[0] += vitesse_actuelle * math.cos(pose_robot[2]) * dt
            pose_robot[1] += vitesse_actuelle * math.sin(pose_robot[2]) * dt
            
            # Mise à jour des vitesses pour détection d'arrêt
            vitesse_precedente = vitesse_actuelle
            vitesse_angulaire_precedente = vitesse_angulaire_corrigee
        
        x_robot, y_robot, angle_robot = pose_robot[0], pose_robot[1], pose_robot[2]
        
        # Si en pause, maintenir l'arrêt
        if est_en_pause:
            driver.setCruisingSpeed(0)
        else:
            # Machine d'état pour le scan
            if mode_scan == "CALIBRATION_GYRO":
                # Calibration du gyroscope
                if temps_actuel < DUREE_CALIBRATION:
                    driver.setCruisingSpeed(0)
                    echantillons_calibration.append(gyro.getValues()[2])
                    print(f"\rCalibration gyro en cours... {temps_actuel:.2f}s", end="")
                    continue
                else:
                    biais_gyro = np.mean(echantillons_calibration)
                    est_calibre = True
                    print(f"\nCalibration terminée. Biais gyro: {biais_gyro:.4f}")
                    mode_scan = "SCAN_INITIAL"
            
            elif mode_scan == "SCAN_INITIAL":
                # Scan initial de référence
                driver.setCruisingSpeed(0)
                driver.setSteeringAngle(0)
                
                if not premier_scan_effectue:
                    effectuer_scan_initial()
                    mode_scan = "DEPLACEMENT"
                    position_debut_deplacement = np.array([x_robot, y_robot])
            
            elif mode_scan == "DEPLACEMENT":
                # Navigation normale
                distances_lidar = nettoyer_donnees_lidar()
                angle_direction = calculer_angle_direction(distances_lidar)
                driver.setSteeringAngle(angle_direction)
                driver.setCruisingSpeed(VITESSE_AUTOPILOTE_M_S)
                
                # Calculer la distance parcourue
                if position_debut_deplacement is not None:
                    distance_parcourue = np.linalg.norm(np.array([x_robot, y_robot]) - position_debut_deplacement)
                    
                    # Si distance atteinte ET véhicule arrêté
                    if distance_parcourue >= DISTANCE_DEPLACEMENT:
                        driver.setCruisingSpeed(0)
                        driver.setSteeringAngle(0)
                        
                        # Vérifier si vraiment arrêté
                        if est_vehicule_arrete(vitesse_precedente, vitesse_angulaire_precedente):
                            mode_scan = "ATTENTE_AVANT_SCAN"
                            temps_arret_complet = temps_actuel
            
            elif mode_scan == "ATTENTE_AVANT_SCAN":
                # Attendre 1.5s après arrêt complet
                driver.setCruisingSpeed(0)
                driver.setSteeringAngle(0)
                
                # Vérifier que le véhicule reste bien arrêté
                if not est_vehicule_arrete(vitesse_precedente, vitesse_angulaire_precedente):
                    # Le véhicule a bougé, retour en déplacement
                    mode_scan = "DEPLACEMENT"
                    temps_arret_complet = None
                elif temps_actuel - temps_arret_complet >= DUREE_ATTENTE_AVANT_SCAN:
                    mode_scan = "SCAN_COMPARAISON"
                    temps_debut_comparaison = temps_actuel
                    scans_comparaison = []
            
            elif mode_scan == "SCAN_COMPARAISON":
                # Scanner pendant 1s et comparer les résultats
                driver.setCruisingSpeed(0)
                driver.setSteeringAngle(0)
                
                # Collecter les scans
                if temps_actuel - temps_debut_comparaison < DUREE_SCAN_COMPARAISON:
                    scan_actuel = obtenir_points_lidar_temps_reel()
                    score = calculer_score_confiance(scan_actuel)
                    scans_comparaison.append((scan_actuel, score))
                else:
                    # Fin de la période de comparaison
                    if scans_comparaison:
                        # Trouver le scan avec le meilleur score
                        meilleur_scan, meilleur_score = max(scans_comparaison, key=lambda x: x[1])
                        
                        # Ajouter à la carte
                        points_carte_globale.extend(meilleur_scan)
                        kdtree_carte = KDTree(points_carte_globale) if points_carte_globale else None
                        
                        print(f"Scan ajouté: {len(meilleur_scan)} points (score: {meilleur_score:.2f})")
                    
                    # Retour en déplacement
                    mode_scan = "DEPLACEMENT"
                    position_debut_deplacement = np.array([x_robot, y_robot])
                    distance_parcourue = 0.0
                    scans_comparaison = []
                    temps_debut_comparaison = None
                    temps_arret_complet = None
        
        # --- Mise à jour des points temps réel ---
        if est_calibre and not est_en_pause:
            points_temps_reel = obtenir_points_lidar_temps_reel()
        
        # --- Mise à jour de l'affichage ---
        if temps_actuel - derniere_mise_a_jour_affichage >= INTERVALLE_AFFICHAGE:
            derniere_mise_a_jour_affichage = temps_actuel
            
            ax_carte.clear()
            ax_carte.set_title(f'Carte SLAM en Temps Réel - Mode: {mode_scan if not est_en_pause else "PAUSE"}')
            ax_carte.grid(True, alpha=0.3)
            ax_carte.set_aspect('equal')
            
            # Points de la carte (vert)
            if points_carte_globale:
                pts = np.array(points_carte_globale)
                ax_carte.scatter(pts[:, 0], pts[:, 1], c='green', s=0.5, alpha=0.6, label='Carte')
            
            # Points temps réel du lidar (bleu)
            if points_temps_reel:
                pts = np.array(points_temps_reel)
                ax_carte.scatter(pts[:, 0], pts[:, 1], c='blue', s=2, alpha=0.8, label='Scan actuel')
            
            # Position et orientation du robot (rouge avec flèche)
            ax_carte.plot(x_robot, y_robot, 'ro', markersize=10, label='Robot')
            longueur_fleche = 0.3
            cos_angle_affichage = math.cos(angle_robot)
            sin_angle_affichage = math.sin(angle_robot)
            ax_carte.arrow(x_robot, y_robot, 
                          longueur_fleche * cos_angle_affichage, longueur_fleche * sin_angle_affichage,
                          head_width=0.15, head_length=0.1, fc='red', ec='red')
            
            # Informations textuelles
            info_texte = f"Temps: {temps_actuel:.1f}s | Position: ({x_robot:.2f}, {y_robot:.2f})m | " \
                        f"Angle: {math.degrees(angle_robot):.1f}° | Points carte: {len(points_carte_globale)}"
            
            if mode_scan == "DEPLACEMENT" and position_debut_deplacement is not None:
                info_texte += f" | Distance: {distance_parcourue:.2f}m"
            elif mode_scan == "SCAN_COMPARAISON" and scans_comparaison:
                scores = [s[1] for s in scans_comparaison]
                if scores:
                    info_texte += f" | Score max: {max(scores):.2f}"
            
            if est_en_pause:
                info_texte += " | EN PAUSE"
            
            ax_carte.text(0.02, 0.98, info_texte, transform=ax_carte.transAxes,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='yellow' if est_en_pause else 'white', alpha=0.8))
            
            ax_carte.set_xlabel('X (m)')
            ax_carte.set_ylabel('Y (m)')
            ax_carte.legend(loc='upper right')
            
            # Ajuster la figure
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        compteur_images += 1

except KeyboardInterrupt:
    print("\nArrêt par l'utilisateur")
except Exception as e:
    print(f"\nErreur: {e}")
    import traceback
    traceback.print_exc()
finally:
    driver.setCruisingSpeed(0)
    plt.ioff()
    if not programme_termine:
        plt.show()
    print("\nCartographie terminée.")
