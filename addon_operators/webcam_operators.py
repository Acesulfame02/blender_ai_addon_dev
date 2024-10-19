import bpy
import cv2
import mediapipe as mp

class AI_OT_WebcamAnimate(bpy.types.Operator):
    bl_idname = "ai.webcam_animate"
    bl_label = "Webcam Animate Rig"
    bl_description = "Animate the rig using webcam input"

    _timer = None
    _cap = None
    _running = False

    def execute(self, context):
        if not AI_OT_WebcamAnimate._running:
            AI_OT_WebcamAnimate._cap = cv2.VideoCapture(0)
            AI_OT_WebcamAnimate._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)
            AI_OT_WebcamAnimate._running = True
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "Webcam animation is already running.")
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'TIMER' and AI_OT_WebcamAnimate._running:
            ret, frame = AI_OT_WebcamAnimate._cap.read()
            if not ret:
                self.cancel(context)
                return {'CANCELLED'}
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp.solutions.pose.Pose().process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [(lmk.x, lmk.y, lmk.z) for lmk in results.pose_landmarks.landmark]
                bpy.app.timers.register(lambda: self.update_rig(context, landmarks), first_interval=0.01)
                bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cancel(context)
                return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        AI_OT_WebcamAnimate._running = False
        AI_OT_WebcamAnimate._cap.release()
        cv2.destroyAllWindows()
        context.window_manager.event_timer_remove(AI_OT_WebcamAnimate._timer)
        return {'CANCELLED'}

    def update_rig(self, context, landmarks):
        try:
            rig = bpy.data.objects[context.scene.rig_object_name]
            if not rig:
                self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")
                return

            # Update arm bones (left and right)
            left_arm_bones = ['shoulder.L', 'elbow.L', 'hand_fk.L']
            right_arm_bones = ['shoulder.R', 'elbow.R', 'hand_fk.R']
            left_landmark_idxs = [11, 13, 15]
            right_landmark_idxs = [12, 14, 16]

            def clamp(value, min_value, max_value):
                return max(min_value, min(value, max_value))

            def update_bone_location(bone, landmark):
                if bone:
                    x = clamp((landmark[0] - 0.5) * 0.1, -0.1, 0.1)
                    y = clamp((landmark[2] - 0.5) * 0.1, -0.1, 0.1)
                    z = clamp((landmark[1] - 0.5) * 0.1, -0.1, 0.1)
                    bone.location = (x, z, -y)
                    bone.keyframe_insert(data_path="location")

            # Update left arm bones
            for i, bone_name in enumerate(left_arm_bones):
                bone = rig.pose.bones.get(bone_name)
                if bone:
                    update_bone_location(bone, landmarks[left_landmark_idxs[i]])

            # Update right arm bones
            for i, bone_name in enumerate(right_arm_bones):
                bone = rig.pose.bones.get(bone_name)
                if bone:
                    update_bone_location(bone, landmarks[right_landmark_idxs[i]])

            bpy.context.view_layer.update()

        except Exception as e:
            print(f"Error updating rig: {e}")

def register():
    bpy.utils.register_class(AI_OT_WebcamAnimate)

def unregister():
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)
