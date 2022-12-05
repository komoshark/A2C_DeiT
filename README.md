
ViT In Actor Critic
=======
Komo Zhang | Dec 2022 | Written in Pytorch

Usage
--------
 * Install required packages : `!pip install -r requirements.txt`
 * To train: `python main.py`
Default working on OpenAI's [Breakout-v4](https://gym.openai.com/envs/Breakout-v4/) environment. To modify this, change env in main.
*  Check logs, picture and model in models folder

Details in Code Folder
--------
*  **Utils . py**  includes data pre-processing and modification of DeiT.
*  **Model . py**  contains model for DeiT_A2C and Simple_A2C.
 *  **Train . py** shows structure for training.


Architecture
--------

```python
self.output = self.DeiT(inputs)
#self.maxlayer = nn.MaxPool2d(kernel_size = 3)
self.critic_linear = nn.Linear(input_num, 1)
self.actor_linear = nn.Linear(input_num, num_actions)
```

\*we use a GRU cell because it has fewer params, uses one memory vector instead of two, and attains the same performance as an LSTM cell.

Environments that work
--------
_(Use `pip freeze` to check your environment settings)_
 * Mac OSX (test mode only) or Linux (train and test)
 * Python 3.6
 * NumPy 1.13.1+
 * Gym 0.9.4+
 * SciPy 0.19.1 (just on two lines -> workarounds possible)
 * [PyTorch 0.4.0](http://pytorch.org/)
