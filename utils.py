import bpy

def apply_keypoints_to_rig(obj, keypoints, frame):
    keypoint_bone_map = {
        "DEF-thigh.L": keypoints[0:3],
        "DEF-shin.L": keypoints[3:6],
        "DEF-foot.L": keypoints[6:9],
        "DEF-thigh.R": keypoints[9:12],
        "DEF-shin.R": keypoints[12:15],
        "DEF-foot.R": keypoints[15:18],
        "DEF-upper_arm.L": keypoints[18:21],
        "DEF-forearm.L": keypoints[21:24],
        "DEF-hand.L": keypoints[24:27],
        "DEF-upper_arm.R": keypoints[27:30],
        "DEF-forearm.R": keypoints[30:33],
        "DEF-hand.R": keypoints[33:36],
    }
    
    for bone_name, keypoint in keypoint_bone_map.items():
        if bone_name in obj.pose.bones:
            bone = obj.pose.bones[bone_name]
            if len(keypoint) == 3:
                bone.location = keypoint
                bone.keyframe_insert(data_path="location", frame=frame)
            else:
                print(f"Warning: Keypoint for {bone_name} does not contain exactly 3 elements: {keypoint}")

    print("Applied keypoints to rig")
