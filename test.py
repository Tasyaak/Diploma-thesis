# import pyhtklib, os, pkgutil

# print("pyhtklib.__file__ =", pyhtklib.__file__)
# print("contents =", os.listdir(os.path.dirname(pyhtklib.__file__)))
# print("submodules =", [m.name for m in pkgutil.iter_modules(pyhtklib.__path__)])


from pyhtklib import Oscilloscope

# Initialize the oscilloscope
osc = Oscilloscope()

# Set up measurement configuration
osc.set_custom_config("measure.config.yaml")

# Initialize and start measurements
if osc.initialize():
    # Collect measurements
    batches = osc.collect_measurements(num_snapshots=4)
    
    # Process and save data
    success, bulk_id = osc.process_data(batches)

    print(success)
    print(bulk_id)
    print(batches)