#!/usr/bin/env python3

"""
Simple test script to verify the ping pong simulation works correctly.
This will load and display the model to check if all components are present.
"""

import mujoco
import mujoco.viewer
import time

def test_model():
    try:
        # Load the ping pong model
        print("Loading ping pong model...")
        model = mujoco.MjModel.from_xml_path('pingpong.xml')
        data = mujoco.MjData(model)
        
        print("Model loaded successfully!")
        print(f"Model name: {model.name}")
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njoint}")
        print(f"Number of geom: {model.ngeom}")
        
        # Print body names
        print("\nBodies in model:")
        for i in range(model.nbody):
            print(f"  {i}: {model.body(i).name}")
            
        # Print joint names  
        print("\nJoints in model:")
        for i in range(model.njoint):
            print(f"  {i}: {model.joint(i).name}")
            
        # Print geom names
        print("\nGeoms in model:")
        for i in range(model.ngeom):
            print(f"  {i}: {model.geom(i).name}")
            
        # Test viewer
        print("\nStarting viewer (press 'q' to quit)...")
        viewer = mujoco.viewer.launch_passive(model, data)
        
        try:
            # Run for a few seconds to see if it works
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < 5:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            viewer.close()
            
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()