import os
import logging
import torch
import tempfile
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from romav2.geometry import pose_auc
import matplotlib.pyplot as plt

OFFSET = 0.5

logger = logging.getLogger(__name__)


class SatAst:
    """
    A benchmark class to evaluate image matchers using a folder of JSON files.

    The final metric is the Area Under the Curve (AUC) of the cumulative
    distribution of *all* individual errors, calculated at different
    pixel thresholds.

    Args:
        json_folder_path (str): Path to the directory containing the .json files.
        image_dataset_root (str): The root directory to prepend to the
        image paths (query_path, pred_path)
        found inside the JSON files.
    """

    def __init__(
        self,
        json_folder_path="data/satast/annotations-v1",
        image_dataset_root="data/satast",
    ) -> None:
        self.json_folder_path = json_folder_path
        self.image_dataset_root = image_dataset_root

        try:
            self.json_files = sorted(
                [f for f in os.listdir(json_folder_path) if f.endswith(".json")]
            )
            if not self.json_files:
                logger.warning("No .json files found in %s", json_folder_path)
        except FileNotFoundError:
            logger.error("Directory not found: %s", json_folder_path)
            self.json_files = []

        logger.info("Found %d JSON files.", len(self.json_files))

    def _pixel_to_normalized(self, pts_pix, w, h, offset=OFFSET):
        """
        Converts pixel coordinates [0, n-1] to normalized [-1, 1].
        """
        pts_norm = np.zeros_like(pts_pix, dtype=np.float64)
        pts_norm[..., 0] = (2.0 * (pts_pix[..., 0] + offset) / w) - 1.0
        pts_norm[..., 1] = (2.0 * (pts_pix[..., 1] + offset) / h) - 1.0
        return pts_norm

    def _normalized_to_pixel(self, pts_norm, w, h, offset=OFFSET):
        """
        Converts normalized coordinates [-1, 1] to pixel [0, n-1].
        """
        pts_pix = np.zeros_like(pts_norm, dtype=np.float64)
        pts_pix[..., 0] = (w * (pts_norm[..., 0] + 1.0) / 2.0) - offset
        pts_pix[..., 1] = (h * (pts_norm[..., 1] + 1.0) / 2.0) - offset
        return pts_pix

    def benchmark(self, model, model_name=None, visualize=False, num_visualize=10):
        """
        Runs the full benchmark on the provided model.
        This version benchmarks correspondences through an estimated homography.

        Args:
            model: The matcher model object. Must have a 'match(im_A_path, im_B_path)'
            method and a 'sample(matches, certainty, num_samples)' method.
            Assumed to return matches in normalized [-1, 1] coordinates.
            model_name (str, optional): A name for the model.
            visualize (bool, optional): If True, plots the first `num_visualize`
            successful homography estimations.
            num_visualize (int, optional): The maximum number of pairs to visualize.

        Returns:
            dict: A dictionary containing the AUC scores.
        """
        # This list will store *all* individual reprojection errors
        all_reprojection_errors = []
        viz_counter = 0
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        im_B_path = temp_file.name

        for json_name in tqdm(self.json_files, desc="Running Benchmark"):
            json_path = os.path.join(self.json_folder_path, json_name)

            # 1. Load JSON and Image Paths
            with open(json_path, "r") as f:
                json_data = json.load(f)

            im_A_path_rel = json_data["query_path"].replace("\\", "/")
            im_B_path_rel = json_data["pred_path"].replace("\\", "/")

            im_A_path = os.path.join(self.image_dataset_root, im_A_path_rel)
            im_B_path0 = os.path.join(self.image_dataset_root, im_B_path_rel)

            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            im_B0 = Image.open(im_B_path0)
            # if "__rot0__" not in im_B_path0:
            #     continue
            w2, h2 = im_B0.size

            for rot_idx in range(1):
                im_B0 = Image.open(im_B_path0)

                if rot_idx == 0:
                    rotated_im_B = im_B0
                elif rot_idx == 1:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_90)
                elif rot_idx == 2:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_180)
                elif rot_idx == 3:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_270)
                rotated_im_B.save(im_B_path, format="JPEG")

                # 2. Run the matcher model
                dense_preds = model.match(im_A_path, im_B_path)
                if isinstance(dense_preds, dict):
                    sparse_preds = model.sample(dense_preds, 10_000)
                else:
                    sparse_preds = model.sample(*dense_preds, 10_000)
                good_matches = sparse_preds[0]
                pos_a_norm = good_matches[:, :2]
                pos_b_norm = good_matches[:, 2:]

                # Rotate back
                if rot_idx == 1:
                    pos_b_norm = torch.stack(
                        [-pos_b_norm[:, 1], pos_b_norm[:, 0]], dim=1
                    )
                elif rot_idx == 2:
                    pos_b_norm = -pos_b_norm
                elif rot_idx == 3:
                    pos_b_norm = torch.stack(
                        [pos_b_norm[:, 1], -pos_b_norm[:, 0]], dim=1
                    )

                # 3. Estimate Homography in Normalized Coordinates
                H_pred_norm = None
                if len(pos_a_norm) >= 4:
                    try:
                        norm_thresh = 0.001

                        H_pred_norm, inliers = cv2.findHomography(
                            pos_a_norm.cpu().numpy(),
                            pos_b_norm.cpu().numpy(),
                            method=cv2.USAC_DEFAULT,
                            confidence=0.99999999,
                            maxIters=100_000,
                            ransacReprojThreshold=norm_thresh,
                        )
                    except cv2.error as e:
                        print(e)
                        H_pred_norm = None

                # 4. Get Ground Truth Correspondences (Pixel)
                last_iteration_corrs = json_data["correspondences"][-1]
                pts_src_pix = np.array(last_iteration_corrs["pts_src"])
                pts_dst_pix = np.array(last_iteration_corrs["pts_dst"])

                if pts_src_pix.shape[0] == 0:
                    continue  # No GT points, skip this pair

                if H_pred_norm is None:
                    # If RANSAC fails, append 'inf' for all GT points
                    all_reprojection_errors.extend(
                        [float("inf")] * pts_src_pix.shape[0]
                    )
                    continue

                if visualize and viz_counter < num_visualize:
                    logger.info("Visualizing pair: %s", json_name)

                    # 1. Create pixel-to-norm (M1) and norm-to-pixel (M2_inv) matrices

                    # M1: P1 -> N1
                    M1 = np.array(
                        [
                            [2.0 / w1, 0, (2.0 * OFFSET / w1) - 1.0],
                            [0, 2.0 / h1, (2.0 * OFFSET / h1) - 1.0],
                            [0, 0, 1.0],
                        ],
                        dtype=np.float64,
                    )

                    # M2_inv: N2 -> P2
                    M2_inv = np.array(
                        [
                            [w2 / 2.0, 0, w2 / 2.0 - OFFSET],
                            [0, h2 / 2.0, h2 / 2.0 - OFFSET],
                            [0, 0, 1.0],
                        ],
                        dtype=np.float64,
                    )

                    # 2. Convert H_pred_norm (N1 -> N2) to H_pix (P1 -> P2)
                    H_pix = M2_inv @ H_pred_norm @ M1

                    # 3. Invert H_pix to get (P2 -> P1) for warping Image 2 to 1
                    try:
                        H_pix_inv = np.linalg.inv(H_pix)
                    except np.linalg.LinAlgError:
                        logger.warning(
                            "Could not invert homography for %s. Skipping visualization.",
                            json_name,
                        )
                        continue  # Skip this viz

                    # --- NEW: Extract and sample inliers for visualization ---
                    plot_pts_a = None
                    plot_pts_b = None
                    plot_colors = None

                    # Get inlier points from the RANSAC
                    inliers_mask = inliers.flatten().astype(bool)
                    inlier_pts_a_norm = pos_a_norm.cpu().numpy()[inliers_mask]
                    inlier_pts_b_norm = pos_b_norm.cpu().numpy()[inliers_mask]
                    num_inliers = inlier_pts_a_norm.shape[0]

                    if num_inliers > 0:
                        # Sample up to 10 inliers
                        num_to_sample = min(num_inliers, 10)
                        indices = np.random.choice(
                            num_inliers, num_to_sample, replace=False
                        )

                        sampled_pts_a_norm = inlier_pts_a_norm[indices]
                        sampled_pts_b_norm = inlier_pts_b_norm[indices]

                        # Convert to pixel coordinates for plotting
                        plot_pts_a = self._normalized_to_pixel(
                            sampled_pts_a_norm, w1, h1
                        )
                        plot_pts_b = self._normalized_to_pixel(
                            sampled_pts_b_norm, w2, h2
                        )

                        # Generate random colors for each pair
                        plot_colors = np.random.rand(num_to_sample, 3)

                    # 4. Load images with OpenCV for warping
                    img1_cv = cv2.imread(im_A_path)  # Target
                    img2_cv = cv2.imread(im_B_path0)  # Source

                    # 5. Warp image 2 to image 1's frame
                    warped_img2 = cv2.warpPerspective(img2_cv, H_pix_inv, (w1, h1))

                    # 6. Plot
                    img1_rgb = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)
                    img2_rgb = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB)
                    warped_img2_rgb = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB)

                    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
                    fig.suptitle(
                        f"Homography Visualization: {os.path.basename(im_B_path0)} -> {os.path.basename(im_A_path)}",
                        fontsize=16,
                    )

                    axes[0].imshow(img1_rgb)
                    axes[0].set_title("Image 1 (Target)")
                    axes[0].axis("off")
                    if plot_pts_a is not None:
                        axes[0].scatter(
                            plot_pts_a[:, 0],
                            plot_pts_a[:, 1],
                            s=40,
                            c=plot_colors,
                            marker="x",
                        )

                    axes[1].imshow(img2_rgb)
                    axes[1].set_title("Image 2 (Source)")
                    axes[1].axis("off")
                    if plot_pts_b is not None:
                        axes[1].scatter(
                            plot_pts_b[:, 0],
                            plot_pts_b[:, 1],
                            s=40,
                            c=plot_colors,
                            marker="x",
                        )

                    axes[2].imshow(warped_img2_rgb)
                    axes[2].set_title("Image 2 Warped to Image 1")
                    axes[2].axis("off")

                    plt.tight_layout()
                    plt.show()

                    viz_counter += 1

                # 5. Normalize GT Source Points
                pts_src_norm = self._pixel_to_normalized(pts_src_pix, w1, h1)

                # 6. Warp Normalized GT Points
                pts_src_norm_h = np.hstack(
                    (pts_src_norm, np.ones((pts_src_norm.shape[0], 1)))
                )
                warped_pts_norm_h = np.dot(pts_src_norm_h, H_pred_norm.T)

                # De-homogenize
                epsilon = 1e-8
                warped_pts_norm = warped_pts_norm_h[:, :2] / (
                    warped_pts_norm_h[:, 2, np.newaxis] + epsilon
                )

                # 7. Convert Warped Points back to Pixel Coords
                warped_pts_pix = self._normalized_to_pixel(warped_pts_norm, w2, h2)

                # 8. Calculate and Store All Errors
                errors = np.linalg.norm(warped_pts_pix - pts_dst_pix, axis=1)
                all_reprojection_errors.extend(errors)

        temp_file.close()

        # 9. Compute AUC over *all* errors
        thresholds = np.arange(1, 31)
        auc = pose_auc(np.array(all_reprojection_errors), thresholds)
        logger.info("=== AUC OF ERRORS ===")
        logger.info("%s", auc)

        results = {
            "reprojection_auc_5px": auc[4],
            "reprojection_auc_10px": auc[9],
            "reprojection_auc_20px": auc[19],
            "reprojection_auc_30px": auc[29],
        }
        logger.info("=== MAIN RESULTS ===")
        logger.info("%s", results)

        return results

    def benchmark_warp(self, model, model_name=None, visualize=False, num_visualize=10):
        """
        Runs the full benchmark on the provided model.
        This version benchmarks correspondences through an estimated dense warp.

        Args:
            model: The matcher model object. Must have a 'match(im_A_path, im_B_path)'
            method and a 'sample(matches, certainty, num_samples)' method.
            Assumed to return matches in normalized [-1, 1] coordinates.
            model_name (str, optional): A name for the model.
            visualize (bool, optional): If True, plots the first `num_visualize`
            successful homography estimations.
            num_visualize (int, optional): The maximum number of pairs to visualize.

        Returns:
            dict: A dictionary containing the AUC scores.
        """
        # This list will store *all* individual reprojection errors
        all_reprojection_errors = []
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        im_B_path = temp_file.name

        for json_name in tqdm(self.json_files, desc="Running Benchmark"):
            json_path = os.path.join(self.json_folder_path, json_name)

            # 1. Load JSON and Image Paths
            with open(json_path, "r") as f:
                json_data = json.load(f)

            im_A_path_rel = json_data["query_path"].replace("\\", "/")
            im_B_path_rel = json_data["pred_path"].replace("\\", "/")

            im_A_path = os.path.join(self.image_dataset_root, im_A_path_rel)
            im_B_path0 = os.path.join(self.image_dataset_root, im_B_path_rel)

            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            im_B0 = Image.open(im_B_path0)
            w2, h2 = im_B0.size

            for rot_idx in range(4):
                im_B0 = Image.open(im_B_path0)

                if rot_idx == 0:
                    rotated_im_B = im_B0
                elif rot_idx == 1:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_90)
                elif rot_idx == 2:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_180)
                elif rot_idx == 3:
                    rotated_im_B = im_B0.transpose(method=Image.Transpose.ROTATE_270)
                rotated_im_B.save(im_B_path, format="JPEG")

                # 2. Run the matcher model
                preds = model.match(im_A_path, im_B_path)
                warp = preds["warp_AB"]

                # 3. Get Ground Truth Correspondences (Pixel)
                last_iteration_corrs = json_data["correspondences"][-1]
                pts_src_pix = np.array(last_iteration_corrs["pts_src"])
                pts_dst_pix = np.array(last_iteration_corrs["pts_dst"])

                if pts_src_pix.shape[0] == 0:
                    continue  # No GT points, skip this pair

                # 5. Normalize GT Source Points
                pts_src_norm = torch.tensor(
                    self._pixel_to_normalized(pts_src_pix, w1, h1),
                    device=warp.device,
                    dtype=warp.dtype,
                )

                # 6. Warp Normalized GT Points
                warp_src_to_pred = warp[0, :, : warp.shape[2] // 2, -2:]
                warped_pts_norm = torch.nn.functional.grid_sample(
                    warp_src_to_pred.permute(2, 0, 1)[None],
                    pts_src_norm[None, None],
                    align_corners=False,
                    mode="bilinear",
                )[0, :, 0].mT

                # Rotate back
                if rot_idx == 1:
                    warped_pts_norm = torch.stack(
                        [-warped_pts_norm[:, 1], warped_pts_norm[:, 0]], dim=1
                    )
                elif rot_idx == 2:
                    warped_pts_norm = -warped_pts_norm
                elif rot_idx == 3:
                    warped_pts_norm = torch.stack(
                        [warped_pts_norm[:, 1], -warped_pts_norm[:, 0]], dim=1
                    )

                # 7. Convert Warped Points back to Pixel Coords
                warped_pts_pix = self._normalized_to_pixel(
                    warped_pts_norm.cpu().numpy(), w2, h2
                )

                # 8. Calculate and Store All Errors
                errors = np.linalg.norm(warped_pts_pix - pts_dst_pix, axis=1)
                all_reprojection_errors.extend(errors)

        temp_file.close()

        # 9. Compute AUC over *all* errors
        thresholds = np.arange(1, 31)
        auc = pose_auc(np.array(all_reprojection_errors), thresholds)
        logger.info("=== AUC OF ERRORS ===")
        logger.info("%s", auc)

        results = {
            "reprojection_auc_5px": auc[4],
            "reprojection_auc_10px": auc[9],
            "reprojection_auc_20px": auc[19],
            "reprojection_auc_30px": auc[29],
        }
        logger.info("=== MAIN RESULTS ===")
        logger.info("%s", results)

        return results
