---
license: cc-by-4.0
task_categories:
- robotics
tags:
- robotics
---
# PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/PppFm7zvXyJblkMOq-YRQ.png)

Github Repo: [Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T)

We provide a set of datasets used for post-training of GR00T N1. Each dataset is a collection of trajectories from different robot embodiments and tasks.

###  Cross-embodied bimanual manipulation: 9k trajectories

| Dataset Name | #trajectories |
| - | -|
| bimanual_panda_gripper.Threading | 1000 |
| bimanual_panda_hand.LiftTray | 1000 |
| bimanual_panda_gripper.ThreePieceAssembly | 1000 |
| bimanual_panda_gripper.Transport | 1000 |
| bimanual_panda_hand.BoxCleanup | 1000 |
| bimanual_panda_hand.DrawerCleanup | 1000 |
| gr1_arms_only.CanSort | 1000 |
| gr1_full_upper_body.Coffee | 1000 |
| gr1_full_upper_body.Pouring | 1000 |


### Humanoid robot tabletop manipulation: 240k trajectories

| Dataset Name | #trajectories |
| - | -|
| gr1_arms_waist.CanToDrawer | 10000 | 
| gr1_arms_waist.CupToDrawer | 10000 | 
| gr1_arms_waist.CuttingboardToBasket | 10000 | 
| gr1_arms_waist.CuttingboardToCardboardBox | 10000 | 
| gr1_arms_waist.CuttingboardToPan | 10000 | 
| gr1_arms_waist.CuttingboardToPot | 10000 | 
| gr1_arms_waist.CuttingboardToTieredBasket | 10000 | 
| gr1_arms_waist.PlaceBottleToCabinet | 10000 | 
| gr1_arms_waist.PlaceMilkToMicrowave | 10000 | 
| gr1_arms_waist.PlacematToBasket | 10000 | 
| gr1_arms_waist.PlacematToBowl | 10000 | 
| gr1_arms_waist.PlacematToPlate | 10000 | 
| gr1_arms_waist.PlacematToTieredShelf | 10000 | 
| gr1_arms_waist.PlateToBowl | 10000 | 
| gr1_arms_waist.PlateToCardboardBox | 10000 | 
| gr1_arms_waist.PlateToPan | 10000 | 
| gr1_arms_waist.PlateToPlate | 10000 | 
| gr1_arms_waist.PotatoToMicrowave | 10000 | 
| gr1_arms_waist.TrayToCardboardBox | 10000 | 
| gr1_arms_waist.TrayToPlate | 10000 | 
| gr1_arms_waist.TrayToPot | 10000 | 
| gr1_arms_waist.TrayToTieredBasket | 10000 | 
| gr1_arms_waist.TrayToTieredShelf | 10000 | 
| gr1_arms_waist.WineToCabinet | 10000 | 

### Humanoid robot tabletop manipulation - downsampled: 24k trajectories

| Dataset Name | #trajectories |
| - | - |
| gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 
| gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000 | 1000 | 

### Robot Arm Kitchen Manipulation: 72K trajectories
| Dataset Name | #trajectories |
| - | -|
| single_panda_gripper.CloseDoubleDoor | 3000 |
| single_panda_gripper.CloseDrawer | 3000 |
| single_panda_gripper.CloseSingleDoor | 3000 |
| single_panda_gripper.CoffeePressButton | 3000 |
| single_panda_gripper.CoffeeServeMug | 3000 |
| single_panda_gripper.CoffeeSetupMug | 3000 |
| single_panda_gripper.OpenDoubleDoor | 3000 |
| single_panda_gripper.OpenDrawer | 3000 |
| single_panda_gripper.OpenSingleDoor | 3000 |
| single_panda_gripper.PnPCabToCounter | 3000 |
| single_panda_gripper.PnPCounterToCab | 3000 |
| single_panda_gripper.PnPCounterToMicrowave | 3000 |
| single_panda_gripper.PnPCounterToSink | 3000 |
| single_panda_gripper.PnPCounterToStove | 3000 |
| single_panda_gripper.PnPMicrowaveToCounter | 3000 |
| single_panda_gripper.PnPSinkToCounter | 3000 |
| single_panda_gripper.PnPStoveToCounter | 3000 |
| single_panda_gripper.TurnOffMicrowave | 3000 |
| single_panda_gripper.TurnOffSinkFaucet | 3000 |
| single_panda_gripper.TurnOffStove | 3000 |
| single_panda_gripper.TurnOnMicrowave | 3000 |
| single_panda_gripper.TurnOnSinkFaucet | 3000 |
| single_panda_gripper.TurnOnStove | 3000 |
| single_panda_gripper.TurnSinkSpout | 3000 |

### Unitree G1 Loco-Manipulation: 102 trajectories
| Dataset Name | #trajectories |
| - | -|
| unitree_g1.LMPnPAppleToPlateDC | 102 |

## Download the dataset from Huggingface

Users can download a specific subset of data by specifying the dataset name.

1. Option 1: with `huggingface-cli`

```bash
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "gr1_arms_only.CanSort/**" \
  --local-dir $HOME/gr00t_dataset
```

**Replace `gr1_arms_only.CanSort/**` with the dataset name you want to download.**

2. Option 2: Github LFS

```bash
# Clone the repo without downloading files
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
cd PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

# Configure sparse-checkout for just your folder
git sparse-checkout init --cone
git sparse-checkout set gr1_arms_waist.CupToDrawer   << Select the specific dataset to download
```

Replace the "gr1_arms_waist.CupToDrawer" with the intended dataset folder
