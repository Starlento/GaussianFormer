_base_ = ['./nuscenes_gs12800.py']

model = dict(
    head=dict(
        SupervisedOnBEV=True
    )
)
