from tensorboard.backend.event_processing import event_accumulator

if __name__ == "__main__":
    
    # Path to your event file or log directory
    #event_path = "/workspace/runs/PNA_drugs_graphcl_123_31-03_15-29-27/events.out.tfevents.1743434967.bb884a644490.1109.0"
    event_path = "/workspace/runs/PNA_drugs_graphcl_123_31-03_15-29-27"
    #event_path = "/workspace/new_Thesis/Thesis/3DInfomax/runs/PNA_ogbg-mollipo_tune_lipo_scaff_3_02-04_16-20-46"
    
    # Initialize the event accumulator
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()  # Loads the events

    # List available tags (like scalar names)
    print("Available tags:", ea.Tags()["scalars"])

    epoch_events = ea.Scalars("epoch/train")
    print(epoch_events)

    #for event in loss_events:
    #    print(f"Step: {event.step}, Value: {event.value}")