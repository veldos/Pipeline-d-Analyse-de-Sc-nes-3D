import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import trimesh

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SIFTMatcher:
    """Classe pour la détection et l'appariement de points SIFT entre images"""
    
    def __init__(self, nfeatures=5000, ratio_threshold=0.7):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.ratio_threshold = ratio_threshold
        self.matcher = cv2.BFMatcher()
    
    def detect_and_compute(self, image):
        """Détecte les points clés et calcule les descripteurs SIFT"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Apparie les descripteurs entre deux images avec le test de ratio de Lowe"""
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches

class PoseEstimator:
    """Classe pour l'estimation de pose entre paires d'images"""
    
    def __init__(self, camera_matrix, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
    
    def estimate_pose_from_matches(self, kp1, kp2, matches, min_matches=50):
        """Estime la pose relative entre deux images à partir des correspondances"""
        if len(matches) < min_matches:
            logger.warning(f"Pas assez de correspondances: {len(matches)} < {min_matches}")
            return None, None
        
        # Extraire les points correspondants
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Calcul de la matrice essentielle
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None, None
        
        # Décomposition de la matrice essentielle
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        return R, t

class StereoReconstructor:
    """Classe pour la reconstruction 3D stéréo"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
    
    def triangulate_points(self, kp1, kp2, matches, R, t):
        """Triangule les points 3D à partir de correspondances stéréo"""
        if len(matches) < 10:
            return np.array([])
        
        # Matrice de projection pour la première caméra (référence)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.camera_matrix @ P1
        
        # Matrice de projection pour la deuxième caméra
        P2 = np.hstack([R, t])
        P2 = self.camera_matrix @ P2
        
        # Points correspondants
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T
        
        # Triangulation
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]  # Conversion homogène vers euclidien
        
        return points_3d.T

class MultiViewReconstructor:
    """Classe principale pour la reconstruction 3D multi-vues"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.sift_matcher = SIFTMatcher()
        self.pose_estimator = PoseEstimator(camera_matrix)
        self.stereo_reconstructor = StereoReconstructor(camera_matrix)
    
    def reconstruct_from_images(self, images):
        """Reconstruit un nuage de points 3D à partir de plusieurs images"""
        if len(images) < 2:
            raise ValueError("Au moins 2 images sont nécessaires pour la reconstruction")
        
        # Détection des caractéristiques pour toutes les images
        keypoints_list = []
        descriptors_list = []
        
        for i, img in enumerate(images):
            logger.info(f"Traitement de l'image {i+1}/{len(images)}")
            kp, desc = self.sift_matcher.detect_and_compute(img)
            keypoints_list.append(kp)
            descriptors_list.append(desc)
        
        # Reconstruction incrémentale
        all_points_3d = []
        
        # Initialisation avec les deux premières images
        matches_01 = self.sift_matcher.match_features(descriptors_list[0], descriptors_list[1])
        if len(matches_01) > 50:
            R, t = self.pose_estimator.estimate_pose_from_matches(
                keypoints_list[0], keypoints_list[1], matches_01
            )
            
            if R is not None and t is not None:
                points_3d = self.stereo_reconstructor.triangulate_points(
                    keypoints_list[0], keypoints_list[1], matches_01, R, t
                )
                
                # Filtrage des points aberrants
                points_3d = self._filter_outliers(points_3d)
                all_points_3d.extend(points_3d)
        
        # Ajout des images suivantes
        for i in range(2, len(images)):
            logger.info(f"Ajout de l'image {i+1} à la reconstruction")
            
            # Correspondances avec l'image de référence
            matches = self.sift_matcher.match_features(descriptors_list[0], descriptors_list[i])
            
            if len(matches) > 30:
                R, t = self.pose_estimator.estimate_pose_from_matches(
                    keypoints_list[0], keypoints_list[i], matches
                )
                
                if R is not None and t is not None:
                    points_3d = self.stereo_reconstructor.triangulate_points(
                        keypoints_list[0], keypoints_list[i], matches, R, t
                    )
                    
                    points_3d = self._filter_outliers(points_3d)
                    all_points_3d.extend(points_3d)
        
        return np.array(all_points_3d) if all_points_3d else np.array([])
    
    def _filter_outliers(self, points_3d, z_threshold=100):
        """Filtre les points aberrants basé sur la distance"""
        if len(points_3d) == 0:
            return points_3d
        
        # Filtrage basé sur la distance à l'origine
        distances = np.linalg.norm(points_3d, axis=1)
        mask = distances < z_threshold
        
        return points_3d[mask]

class PointNetClassifier(nn.Module):
    """Implémentation de PointNet pour la classification de nuages de points"""
    
    def __init__(self, num_classes=10, num_points=1024):
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        
        # Transformation spatiale
        self.stn3d = STN3d()
        
        # MLP partagé
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Transformation des caractéristiques
        self.fstn = STNkd(k=64)
        
        # Couches de classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        B, D, N = x.size()
        
        # Transformation spatiale 3D
        trans = self.stn3d(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Transformation des caractéristiques
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        # MLP (128, 1024)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling global
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Classification MLP
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class STN3d(nn.Module):
    """Réseau de transformation spatiale 3D"""
    
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    """Réseau de transformation spatiale k-dimensionnel"""
    
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(self.k, dtype=torch.float32, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class ModelNet10Dataset(Dataset):
    """Dataset pour ModelNet10"""
    
    def __init__(self, data_dir, split='train', num_points=1024, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.transform = transform
        
        # Classes ModelNet10
        self.classes = [
            'bathtub', 'bed', 'chair', 'desk', 'dresser',
            'monitor', 'night_stand', 'sofa', 'table', 'toilet'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Charger les fichiers
        self.files = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name / split
            if class_dir.exists():
                for file in class_dir.glob('*.off'):
                    self.files.append(file)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Charger le mesh
        mesh_path = self.files[idx]
        label = self.labels[idx]
        
        # Charger avec trimesh
        mesh = trimesh.load(str(mesh_path))
        
        # Échantillonner des points sur la surface
        if hasattr(mesh, 'sample'):
            points = mesh.sample(self.num_points)
        else:
            # Fallback: utiliser les vertices si l'échantillonnage échoue
            vertices = np.array(mesh.vertices)
            if len(vertices) >= self.num_points:
                idx_sample = np.random.choice(len(vertices), self.num_points, replace=False)
                points = vertices[idx_sample]
            else:
                # Répliquer les points si pas assez
                idx_sample = np.random.choice(len(vertices), self.num_points, replace=True)
                points = vertices[idx_sample]
        
        # Normalisation
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))
        
        # Application des transformations
        if self.transform:
            points = self.transform(points)
        
        return torch.FloatTensor(points).transpose(1, 0), label

class SceneAnalysisPipeline:
    """Pipeline principal d'analyse de scènes 3D"""
    
    def __init__(self, camera_matrix, model_path=None):
        self.camera_matrix = camera_matrix
        self.reconstructor = MultiViewReconstructor(camera_matrix)
        
        # Charger ou initialiser le modèle PointNet
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pointnet = PointNetClassifier(num_classes=10).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.pointnet.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Modèle chargé depuis {model_path}")
        
        self.classes = [
            'bathtub', 'bed', 'chair', 'desk', 'dresser',
            'monitor', 'night_stand', 'sofa', 'table', 'toilet'
        ]
    
    def train_pointnet(self, data_dir, epochs=50, batch_size=32, lr=0.001):
        """Entraîne le modèle PointNet sur ModelNet10"""
        logger.info("Début de l'entraînement du modèle PointNet")
        
        # Datasets
        train_dataset = ModelNet10Dataset(data_dir, split='train')
        test_dataset = ModelNet10Dataset(data_dir, split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimiseur et fonction de perte
        optimizer = torch.optim.Adam(self.pointnet.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Entraînement
            self.pointnet.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.pointnet(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += len(target)
            
            # Validation
            self.pointnet.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.pointnet(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += len(target)
            
            train_acc = 100.0 * train_correct / train_total
            test_acc = 100.0 * test_correct / test_total
            
            logger.info(f'Epoch {epoch+1}/{epochs}: '
                       f'Train Loss: {train_loss/len(train_loader):.4f}, '
                       f'Train Acc: {train_acc:.2f}%, '
                       f'Test Acc: {test_acc:.2f}%')
            
            # Sauvegarde du meilleur modèle
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.pointnet.state_dict(), 'best_pointnet_model.pth')
                logger.info(f'Nouveau meilleur modèle sauvegardé avec {test_acc:.2f}% de précision')
    
    def process_scene(self, images):
        """Traite une scène complète: reconstruction 3D + classification"""
        logger.info("Début du traitement de la scène")
        
        # Étape 1: Reconstruction 3D
        logger.info("Reconstruction 3D en cours...")
        points_3d = self.reconstructor.reconstruct_from_images(images)
        
        if len(points_3d) == 0:
            logger.error("Échec de la reconstruction 3D")
            return None, None
        
        logger.info(f"Reconstruction réussie: {len(points_3d)} points 3D générés")
        
        # Étape 2: Préparation pour la classification
        point_cloud = self._prepare_point_cloud(points_3d)
        
        # Étape 3: Classification
        logger.info("Classification en cours...")
        prediction, confidence = self._classify_point_cloud(point_cloud)
        
        return point_cloud, {
            'class': self.classes[prediction],
            'confidence': confidence,
            'class_id': prediction
        }
    
    def _prepare_point_cloud(self, points_3d, num_points=1024):
        """Prépare le nuage de points pour la classification"""
        # Échantillonnage ou duplication pour avoir exactement num_points
        if len(points_3d) >= num_points:
            idx = np.random.choice(len(points_3d), num_points, replace=False)
            sampled_points = points_3d[idx]
        else:
            idx = np.random.choice(len(points_3d), num_points, replace=True)
            sampled_points = points_3d[idx]
        
        # Normalisation
        centroid = np.mean(sampled_points, axis=0)
        sampled_points = sampled_points - centroid
        
        max_dist = np.max(np.linalg.norm(sampled_points, axis=1))
        if max_dist > 0:
            sampled_points = sampled_points / max_dist
        
        return sampled_points
    
    def _classify_point_cloud(self, point_cloud):
        """Classifie un nuage de points avec PointNet"""
        self.pointnet.eval()
        
        # Conversion en tensor PyTorch
        points_tensor = torch.FloatTensor(point_cloud).unsqueeze(0).transpose(2, 1).to(self.device)
        
        with torch.no_grad():
            output = self.pointnet(points_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    
    def visualize_reconstruction(self, point_cloud, prediction_info=None):
        """Visualise le nuage de points reconstruit"""
        if len(point_cloud) == 0:
            logger.warning("Aucun point à visualiser")
            return
        
        # Créer un nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Coloration basée sur la hauteur (axe Z)
        colors = plt.cm.viridis((np.array(point_cloud)[:, 2] - np.min(point_cloud[:, 2])) / 
                               (np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Affichage
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Reconstruction 3D")
        vis.add_geometry(pcd)
        
        if prediction_info:
            logger.info(f"Classe prédite: {prediction_info['class']} "
                       f"(confiance: {prediction_info['confidence']:.2%})")
        
        vis.run()
        vis.destroy_window()

# Fonction utilitaire pour créer une matrice de calibration par défaut
def create_default_camera_matrix(image_width=640, image_height=480):
    """Crée une matrice de calibration de caméra par défaut"""
    focal_length = max(image_width, image_height)
    camera_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return camera_matrix

# Exemple d'utilisation
def example_usage():
    """Exemple d'utilisation du pipeline"""
    logger.info("Démarrage de l'exemple d'utilisation")
    
    # Configuration de la caméra
    camera_matrix = create_default_camera_matrix()
    
    # Initialisation du pipeline
    pipeline = SceneAnalysisPipeline(camera_matrix)
    
    # Exemple avec des images synthétiques (à remplacer par de vraies images)
    logger.info("Création d'images d'exemple")
    images = []
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Ajouter quelques caractéristiques détectables
        cv2.circle(img, (200 + i*50, 200), 30, (255, 255, 255), -1)
        cv2.rectangle(img, (300 + i*20, 150), (400 + i*20, 250), (0, 255, 0), -1)
        images.append(img)
    
    # Traitement de la scène
    try:
        point_cloud, prediction = pipeline.process_scene(images)
        
        if point_cloud is not None:
            logger.info(f"Reconstruction réussie avec {len(point_cloud)} points")
            
            if prediction:
                logger.info(f"Classification: {prediction['class']} "
                           f"(confiance: {prediction['confidence']:.2%})")
            
            # Visualisation (optionnelle)
            # pipeline.visualize_reconstruction(point_cloud, prediction)
        else:
            logger.error("Échec du traitement de la scène")
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")

if __name__ == "__main__":
    example_usage()
