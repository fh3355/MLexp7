from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import os
import csv
import logging

# 设置日志记录器
logging.basicConfig(
    filename='./MLexp7.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import jupyter
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from matplotlib import rcParams
from matplotlib import font_manager
font_path = './data/simhei.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False
logging.debug(plt.rcParams['font.family'])

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    )

# 其中normalizeString函数中的正则表达式需对应更改，否则会将中文单词替换成空格
def normalizeString(s):
    # 小写并去掉前后空格
    s = s.lower().strip()

    # 如果句子中没有空格（常见中文句子），按字分开
    if ' ' not in s:
        s = list(s)
        s = ' '.join(s)
    # 不保留标点符号：移除常见英文标点和中文标点
    # 英文标点使用 string.punctuation，中文标点列举常见字符
    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", s)
    s = re.sub(r"[。！？，、；：”“‘’（）《》——…·『』『』]", "", s)

    # 合并多余空格并返回
    s = re.sub(r"\s+", ' ', s).strip()
    return s

def readLangs(lang1, lang2, reverse=False):
    logging.debug("Reading lines...")

    # Read the file and split into lines
    file_path = "./data/eng-cmn.txt"
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
    

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]


    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    logging.info(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    logging.info(f"Trimmed to {len(pairs)} sentence pairs")
    logging.info("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logging.info("Counted words:")
    logging.info(f'{input_lang.name}  {input_lang.n_words}')
    logging.info(f'{output_lang.name}  {output_lang.n_words}')
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
logging.info(f'{random.choice(pairs)}')
# 输出

'''file_path = "./data/eng-cmn.txt"
with open(file_path, encoding='utf-8') as file:
    lines = file.readlines()
pairs = [[normalizeString(l).split('\t')[:2]] for l in lines]
cn = []
eng = []
for p in pairs:
    p=np.array(p)
    eng.append([p[0,0]])
    cn.append([p[0,1]])'''

# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 注意力解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Preparing Training Data
def indexesFromSentence(lang, sentence):
    # split on any whitespace and skip empty tokens to avoid KeyError on ''
    indexes = []
    for word in sentence.strip().split():
        if not word:
            continue
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            # Unknown token: log a warning and skip it (consider adding UNK support later)
            logging.warning(f"Unknown token '{word}' for language {lang.name}; skipping")
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# Training the model
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

#This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate_bleu(encoder, decoder, n=100):
    """Compute average BLEU over n random pairs without printing."""
    with torch.no_grad():
        sum_scores = 0.0
        for i in range(n):
            pair = random.choice(pairs)
            output_words, _ = evaluate(encoder, decoder, pair[0])
            # reference: list of tokens
            ref = pair[1].strip().split(' ')
            ref.append('<EOS>')
            try:
                bleu = sentence_bleu([ref], output_words)
            except Exception:
                bleu = 0.0
            sum_scores += bleu
        return sum_scores / n

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01, save_every=0, eval_every=0, eval_n=100, model_dir='model', result_dir='result'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    loss_history = []

    # CSV headers for history files
    loss_csv = os.path.join(result_dir, 'loss_history.csv')
    bleu_csv = os.path.join(result_dir, 'bleu_history.csv')
    # create/overwrite CSV files with headers
    with open(loss_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'loss'])
    with open(bleu_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'bleu'])

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        loss_history.append((iter, loss))

        # append per-iteration loss to CSV (optional heavy I/O, but safe)
        with open(loss_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter, loss])

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            msg = '%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg)
            print(msg)
            logging.info(msg)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # Periodic evaluation (BLEU) and model saving if configured
        if save_every and iter % save_every == 0:
            # save model weights
            torch.save(encoder.state_dict(), os.path.join(model_dir, f'encoder_iter{iter}.pth'))
            torch.save(decoder.state_dict(), os.path.join(model_dir, f'decoder_iter{iter}.pth'))
            logging.info(f"Saved models at iter {iter} to {model_dir}")

        if eval_every and iter % eval_every == 0:
            # compute BLEU on random samples and append to CSV
            try:
                bleu_avg = evaluate_bleu(encoder, decoder, n=eval_n)
            except Exception as e:
                bleu_avg = 0.0
                logging.exception('Error computing BLEU:')
            with open(bleu_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iter, bleu_avg])
            logging.info(f"Iter {iter}: BLEU={bleu_avg}")

    # 保存最终的模型与完整loss
    torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder_final.pth'))
    torch.save(decoder.state_dict(), os.path.join(model_dir, 'decoder_final.pth'))

    # 完整 loss 文件（冗余，但方便一次性读取）
    final_loss_csv = os.path.join(model_dir, 'loss_full.csv')
    with open(final_loss_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'loss'])
        for it, lv in loss_history:
            writer.writerow([it, lv])

    # save loss plot to model_dir
    try:
        showPlot(plot_losses, os.path.join(result_dir, 'loss_plot.png'))
        logging.info(f"Saved loss plot to {os.path.join(model_dir, 'loss_plot.png')}")
    except Exception:
        # fallback to showing if saving failed
        logging.exception('Failed to save loss plot')
        showPlot(plot_losses)

# Plotting results
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points, out_path=None):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    # If no out_path provided, save to default models/loss_plot.png
    if out_path is None:
        os.makedirs('models', exist_ok=True)
        out_path = os.path.join('models', 'loss_plot.png')
    try:
        plt.savefig(out_path)
        plt.close()
        logging.info(f"Saved plot to {out_path}")
    except Exception:
        logging.exception(f"Failed to save plot to {out_path}")
        # fallback: try to show (only if save totally fails)
        try:
            plt.show()
        except Exception:
            logging.exception('Failed to show plot as fallback')

# Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
def evaluateRandomly(encoder, decoder, n=100):
    sum_scores = 0
    for i in range(n):
        pair = random.choice(pairs)
        logging.info(f'>  {pair[0]}')
        logging.info(f'=  {pair[1]}')
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        logging.info(f'<  {output_sentence}')
        logging.info('\n')
        w = []
        words = pair[1].strip(' ').split(' ')
        words.append('<EOS>')
        w.append(words)
        bleu_score = sentence_bleu(w, output_words)
        sum_scores += bleu_score
    logging.info(f'The bleu_score is {sum_scores/n}')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # save attention figure to result/ with timestamp and the first 4 characters of the input
    os.makedirs('result', exist_ok=True)
    inputtag = input_sentence.replace(" ", "")[:4]
    fname = f"attention_{int(time.time())}_{inputtag}.png"
    out_path = os.path.join('result', fname)
    try:
        plt.savefig(out_path)
        plt.close()
        logging.info(f"Saved attention visualization to {out_path}")
    except Exception:
        logging.exception('Failed to save attention visualization')
        plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    logging.info(f'input = {input_sentence}')
    logging.info(f'output = {" ".join(output_words)}')
    showAttention(input_sentence, output_words, attentions)

# Training and Evaluating
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, n_iters=75000, print_every=5000, eval_every=100)
# 输出

evaluateRandomly(encoder1, attn_decoder1)
# 输出

# plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "你 只 是 玩")
# logging.info(f'{output_words}')
# os.makedirs('models', exist_ok=True)
# attn_path = os.path.join('models', 'attentions_example_1.png')
# plt.matshow(attentions.numpy())
# try:
#     plt.savefig(attn_path)
#     plt.close()
#     logging.info(f"Saved attention matrix to {attn_path}")
# except Exception:
#     logging.exception('Failed to save attention example image')
#     plt.show()
evaluateAndShowAttention("你 只 是 玩")
# 输出


evaluateAndShowAttention("他 和 他 的 邻 居 相 处 ")

evaluateAndShowAttention("我 肯 定 他 会 成 功 的 ")

evaluateAndShowAttention("他 總 是 忘 記 事 情")

evaluateAndShowAttention("我 们 非 常 需 要 食 物 ")
# 输出