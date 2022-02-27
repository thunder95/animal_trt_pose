import paddle
from dataset import AnimalPoseDataset
from trt_pose_model import get_model
import json
import numpy as np

paddle.disable_static()
paddle.device.set_device("gpu")

def save_log(msg):
    fp=open("train.log","a",encoding="utf-8")
    fp.write(msg+"\n")
    fp.close()

with open("animal_pose.json", "r") as f:
    pose_config = json.load(f)
num_parts = len(pose_config["keypoints"])
num_links = len(pose_config["skeleton"])
model = get_model(num_parts, 2 * num_links)

#是否加载预训练模型
# wgts = paddle.load("/home/aistudio/work/human/trt_pose.pdparams")
# wgts = paddle.load("/home/aistudio/work/animal/2_trtpose.pdparams")
# model.set_state_dict(wgts)

scheduler = paddle.optimizer.lr.StepDecay(learning_rate=1e-3, step_size=30, gamma=0.1, verbose=True)
optimizer= paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

train_dataset = AnimalPoseDataset(data_path = "/home/aistudio/data", mode="train")
train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False)

test_dataset = AnimalPoseDataset(data_path = "/home/aistudio/data", mode="test")
test_dataloader = paddle.io.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False)

print("data amount: ", len(train_dataloader), len(test_dataloader))

EPOCHS = 100
loss_train = []
loss_eval = []
min_loss=1e6
for epoch in range(EPOCHS):
    model.train()
    for step, (image, cmap, paf, mask) in enumerate(train_dataloader):
        cmap_out, paf_out = model(image)
        cmap_mse = paddle.mean(mask * (cmap_out - cmap)**2)
        paf_mse = paddle.mean(mask * (paf_out - paf)**2) 
        loss = cmap_mse + paf_mse
        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()

        if step % 10 == 0:
            msg_str = "step: %d, cmap loss: %.5f, paf loss: %.5f, total loss: %.5f, lr: %.5f" % ( 
                step, cmap_mse.numpy()[0], paf_mse.numpy()[0], loss.numpy()[0], scheduler.get_lr())
            print(msg_str)
            save_log(msg_str)
            loss_train.append(loss.numpy()[0])

    scheduler.step()

    model.eval()
    test_loss = 0.0
    for step, (image, cmap, paf, mask) in enumerate(test_dataloader):
        cmap_out, paf_out = model(image)
        cmap_mse = paddle.mean(mask * (cmap_out - cmap)**2)
        paf_mse = paddle.mean(mask * (paf_out - paf)**2) 
        loss = cmap_mse + paf_mse
        test_loss += loss.numpy()[0]
    test_loss /= len(test_dataloader)
    msg_str = "epoch: %d, eval loss: %.5f" % (epoch, test_loss)
    print(msg_str)
    save_log(msg_str)
    loss_eval.append(test_loss)

    if epoch % 10 == 0:
        paddle.save(model.state_dict(), str(epoch)+"_trtpose.pdparams")

    if test_loss < min_loss:
        paddle.save(model.state_dict(), "best_trtpose.pdparams")
        min_loss = test_loss

    np.save("train_loss", np.asarray(loss_train))
    np.save("eval_loss", np.asarray(loss_eval))


        
