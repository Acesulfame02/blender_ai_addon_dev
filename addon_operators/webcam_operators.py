import bpy
import cv2
import mediapipe as mp

from ..utilfunc.utils import update_rig_with_landmarks

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
            hands = mp.solutions.hands.Hands()
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = hand_info.classification[0].label
                    rig = bpy.data.objects.get(context.scene.rig_object_name)
                    if rig:
                        update_rig_with_landmarks(rig, hand_landmarks, hand_type)  # Delegate to utils.py for rig updating
                    else:
                        self.report({'ERROR'}, f"Rig object '{context.scene.rig_object_name}' not found")

            cv2.imshow('Mediapipe Hands', frame)

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

def register():
    bpy.utils.register_class(AI_OT_WebcamAnimate)

def unregister():
    bpy.utils.unregister_class(AI_OT_WebcamAnimate)
