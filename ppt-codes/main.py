import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn
from transformers import ViTFeatureExtractor
from nltk.translate.bleu_score import corpus_bleu

from data import Dictionary,AverageMeter,batchfy
from utils import build_words_dict,get_images,split_dataset
from model import DecoderWithAttention,Encoder
from data import Logger,CaptionDataset
from utils import adjust_learning_rate,clip_gradient
from evaluate import accuracy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default="./data",type=str)
    parser.add_argument('--result-dir',default="./result",type=str)
    parser.add_argument('--attention-dim',default=512,type=int)
    parser.add_argument('--emb-dim',default=512,type=int)
    parser.add_argument('--decoder-dim',default=512,type=int)
    parser.add_argument('--dropout',default=0.5,type=float)
    parser.add_argument('--decoder-lr',default=4e-4,type=float)
    parser.add_argument('--encoder-lr',default=1e-4,type=float)
    parser.add_argument('--fine-tune-encoder',action='store_true')
    parser.add_argument('--dataset',default='wukong',type=str)
    parser.add_argument('--checkpoint',default=None,type=str)
    parser.add_argument('--batch-size',default=64,type=int)
    parser.add_argument('--workers',default=1,type=int)
    parser.add_argument('--epochs',default=120,type=int)
    parser.add_argument('--alpha-c',default=1.0,type=float)
    # regularization parameter for 'doubly stochastic attention', as in the paper
    parser.add_argument('--grad-clip',default=5.0,type=float)
    parser.add_argument('--print-freq',default=1,type=int)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    data_dataset_dir = os.path.join(args.data_dir,args.dataset)
    result_dataset_dir = os.path.join(args.result_dir,args.dataset)
    if not os.path.exists(result_dataset_dir):
        os.makedirs(result_dataset_dir)
    log_file = os.path.join(result_dataset_dir,"images.log")
    logger = Logger(log_file)
    save_word_file = os.path.join(result_dataset_dir,"words.json")
    load_word_file = os.path.join(data_dataset_dir,"chinese_dict.txt")
    
    if not os.path.exists(save_word_file):
        build_words_dict(load_word_file,save_word_file)
    logger.info("file saved in file path: %s"%save_word_file)
    load_csv_file = os.path.join(data_dataset_dir,"wukong50k_release.csv")
    save_data_path = os.path.join(result_dataset_dir,"raw")
    # download the dataset
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
        get_images(load_csv_file,save_data_path)
    elif len(os.listdir(save_data_path))==0:
        get_images(load_csv_file,save_data_path)
    else:
        pass
    logger.info("images saved in file path: %s"%save_data_path)
    save_split_path = os.path.join(result_dataset_dir,"processed")
    if not os.path.exists(save_split_path):
        os.makedirs(save_split_path)
        split_dataset(save_data_path,save_split_path)
    logger.info("images saved in file path: %s"%save_split_path)
    
    word_dict = Dictionary.load(save_word_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    # Initialize / load checkpoint
    best_bleu4 = 0.0
    if args.checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=len(word_dict),
                                       dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder = Encoder().to(device)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None
        start_epoch = 1
        epochs_since_improvement = 0 

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Custom dataloaders
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((640,480)),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    transform = torchvision.transforms.Resize((640,480))
    # transform =torchvision.transforms.CenterCrop((640,480)) 
    train_dataset = CaptionDataset(result_dataset_dir,'train',word_dict,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, 
                                               collate_fn=batchfy,
                                               pin_memory=True)
    valid_dataset = CaptionDataset(result_dataset_dir,'valid',word_dict,transform=transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               collate_fn=batchfy,
                                               pin_memory=True)
    test_dataset = CaptionDataset(result_dataset_dir,'test',word_dict,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers,
                                              collate_fn=batchfy,
                                              pin_memory=True)
    train_loss_list = []
    train_top5score = []
    valid_loss_list = []
    valid_top5score = []
    valid_blue4_list = []
    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        losses,top5accs = train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              device=device,
              alpha_c = args.alpha_c,
              grad_clip = args.grad_clip,
              print_freq = args.print_freq)
        train_top5score.append(top5accs)
        train_loss_list.append(losses)
        # One epoch's validation
        recent_bleu4,top5accs,losses = validate(val_loader=valid_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_dict=word_dict,
                                alpha_c=args.alpha_c,
                                device=device,
                                print_freq=args.print_freq)
        valid_blue4_list.append(recent_bleu4)
        valid_loss_list.append(losses)
        valid_top5score.append(top5accs)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.dataset, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
    save_list(train_loss_list,'train-loss.txt')
    save_list(train_top5score,'train-score.txt')
    save_list(valid_loss_list,'valid-loss.txt')
    save_list(valid_top5score,'valid-score.txt')
    save_list(valid_blue4_list,'valid-blue4.txt')
def save_list(data_list,file_name):
    with open(file_name,mode='w',encoding='utf-8') as wfp:
        for value in data_list:
            wfp.write(str(value)+"\n")
def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,device,alpha_c,
          grad_clip,print_freq):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    # fea_extract = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    # Batches
    for idx,item in enumerate(train_loader):
        index,sent,imgs,caplens = item
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        sent = sent.to(device)

        # Forward prop.
        # imgs = fea_extract(imgs, return_tensors="pt").to(device)
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs,sent,caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # len_sent = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # print(len(len_sent))
        # print(len_sent)
        # exit()
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, idx, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return losses.get_avg(),top5accs.get_avg()
def validate(val_loader, encoder, decoder, criterion,word_dict,alpha_c,device,print_freq,):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for idx,item in enumerate(val_loader):
            index,sent,imgs,caplens = item
            # Move to device, if available
            imgs = imgs.to(device)
            sent = sent.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, sent, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if idx % print_freq == 0:
                print('Validation: [{0}/{1}] '
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f}) '.format(idx, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = sent[sort_ind]  # because images were sorted in the decoder
            allcaps = allcaps.repeat(allcaps.shape[0],1,1)
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_dict.start,word_dict.pad}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4,top5accs,losses

if __name__ == "__main__":
    main()
