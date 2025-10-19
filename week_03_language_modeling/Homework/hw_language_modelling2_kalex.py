import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import nltk

from collections import Counter
from typing import List

import seaborn


# Получить отдельные предложения и поместить их в sentences

def preprocess_data(dataset, word_threshold=50, vocab_size=10000):
    """
    Препроцессинг данных и создание словаря

    Args:
        dataset: Загруженный датасет IMDB
        word_threshold: Максимальное количество слов в предложении
        vocab_size: Размер словаря (без учета служебных токенов)

    Returns:
        vocab: Множество слов (словарь)
        sentences: Список предложений после препроцессинга
        word_freq: Распределение частот слов
    """

    all_sentences = []

    print("🎯 Шаг 1/4: Разделение текстов на предложения и фильтрация по длине")

    # 1. Разделяем на предложения и фильтруем по длине
    # for split in ['train', 'test']:
    for split in ['train']:
        print(f"\n📂 Обработка {split} данных...")

        # Прогресс бар для каждого сплита
        for text in tqdm(dataset[split]['text'], desc=f"Обработка {split}"):
            # Разделяем текст на предложения
            sentences = sent_tokenize(text)

            for sentence in sentences:
                # Токенизируем предложение на слова
                words = word_tokenize(sentence)

                # 2. Оставляем только предложения с количеством слов < word_threshold
                if len(words) < word_threshold:
                    all_sentences.append(sentence)

    print(f"✅ Всего предложений после фильтрации: {len(all_sentences):,}")

    # 3. Считаем частоту вхождения каждого слова
    print(f"\n🎯 Шаг 2/4: Подсчет частот слов")
    word_counter = Counter()

    with tqdm(total=len(all_sentences), desc="Токенизация предложений") as pbar:
        for sentence in all_sentences:
            words = word_tokenize(sentence.lower())  # приводим к нижнему регистру
            word_counter.update(words)
            pbar.update(1)

    print(f"✅ Всего уникальных слов: {len(word_counter):,}")
    print(f"📊 10 самых частых слов: {word_counter.most_common(10)}")

    # 4. Создаем словарь
    print(f"\n🎯 Шаг 3/4: Создание словаря")
    # Формируем упорядоченный словарь: специальные токены + частотные слова
    special_tokens = ['<unk>', '<bos>', '<eos>', '<pad>']
    print(f"✅ Добавлены служебные токены: {special_tokens}")

    print(f"🎯 Шаг 4/4: Добавление {vocab_size} самых частых слов в словарь")
    most_common_words = [word for word, count in tqdm(
        word_counter.most_common(vocab_size),
        desc="Формирование словаря",
        total=vocab_size
    )]

    vocab_list = special_tokens + most_common_words

    print(f"✅ Размер итогового словаря: {len(vocab_list):,}")

    return vocab_list, all_sentences, word_counter

def collate_fn_with_padding(pad_id: int):
    def _collate(input_batch: List[List[int]]) -> torch.Tensor:
        seq_lens = [len(x) for x in input_batch]
        max_seq_len = max(seq_lens)

        new_batch = []
        for sequence in input_batch:
            for _ in range(max_seq_len - len(sequence)):
                sequence.append(pad_id)
            new_batch.append(sequence)

        sequences = torch.LongTensor(new_batch).to(get_device())

        new_batch = {
            'input_ids': sequences[:,:-1],
            'target_ids': sequences[:,1:]
        }

        return new_batch
    return _collate

class WordDataset(Dataset):
    def __init__(self, sentences, word2ind):
        """
        Args:
            sentences: Список предложений
            word2ind: Словарь для преобразования слов в индексы
        """
        self.data = sentences
        self.word2ind = word2ind
        self.unk_id = word2ind['<unk>']
        self.bos_id = word2ind['<bos>']
        self.eos_id = word2ind['<eos>']
        self.pad_id = word2ind['<pad>']

        print(f"📊 Инициализирован WordDataset с {len(self.data)} предложениями")

    def __getitem__(self, idx: int) -> List[int]:
        """
        Возвращает последовательность индексов для предложения

        Args:
            idx: Индекс предложения

        Returns:
            List[int]: Список индексов с <bos> и <eos> токенами
        """
        sentence = self.data[idx]

        # Токенизируем предложение
        words = word_tokenize(sentence.lower())

        # Создаем последовательность индексов
        tokenized_sentence = []

        # Добавляем токен начала последовательности <bos>
        tokenized_sentence.append(self.bos_id)

        # Добавляем индексы слов
        for word in words:
            # Если слова нет в словаре, используем <unk>
            word_id = self.word2ind.get(word, self.unk_id)
            tokenized_sentence.append(word_id)

        # Добавляем токен конца последовательности <eos>
        tokenized_sentence.append(self.eos_id)

        return tokenized_sentence

    def __len__(self) -> int:
        return len(self.data)

    def get_example_info(self, idx: int):
        """Вспомогательная функция для просмотра примера"""
        sentence = self.data[idx]
        indices = self[idx]

        # Обратное преобразование для демонстрации
        ind2word = {v: k for k, v in self.word2ind.items()}
        words_reconstructed = [ind2word.get(i, f'<unk_{i}>') for i in indices]

        return {
            'original_sentence': sentence,
            'indices': indices,
            'reconstructed': ' '.join(words_reconstructed),
            'length': len(indices)
        }


import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import math

class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_batch: torch.Tensor, hidden_state: torch.Tensor = None) -> torch.Tensor:
        # input_batch shape: (batch_size, seq_length)

        # Эмбеддинги
        embedded = self.embedding(input_batch)

        # GRU с возможностью передачи начального скрытого состояния
        if hidden_state is not None:
            gru_output, hidden_state = self.gru(embedded, hidden_state)
        else:
            gru_output, hidden_state = self.gru(embedded)

        # Регуляризация и выход
        gru_output = self.dropout(gru_output)
        logits = self.output_layer(gru_output)

        return logits

    def __calc_loss(self, batch):
      logits = self.__model(batch['input_ids'])
      loss = self.__criterion(logits, batch['target_ids'].flatten())
      return loss

    def __calc_pp(self, dataloader) -> float:
        self.__model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                losses.append(self.__calc_loss(batch).item())

        perplexity = np.exp(np.mean(losses))

        return perplexity


    def init_hidden1(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния"""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)



def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        vocab_size: int,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        device: torch.device = None,
        pad_id: int = 0
) -> dict:
    # Определяем устройство
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # ignore padding tokens
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Для отслеживания прогресса
    train_losses = []
    eval_perplexities = []
    learning_rates = []

    print(f"Training on: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # Режим обучения
        model.train()
        epoch_train_loss = 0.0
        total_tokens = 0

        # Progress bar для обучения
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for batch_idx, batch in enumerate(train_bar):
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            if hasattr(model, 'init_hidden'):
                # Для моделей с возможностью инициализации скрытого состояния
                hidden = model.init_hidden(input_ids.size(0), device)
                logits, _ = model(input_ids, hidden)
            else:
                logits = model(input_ids)

            # Подготавливаем данные для вычисления потерь
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = target_ids.reshape(-1)

            # Вычисляем потери
            loss = criterion(logits_flat, targets_flat)

            # Обратный проход
            loss.backward()

            # Обновляем веса
            optimizer.step()

            # Собираем статистику
            batch_tokens = targets_flat.ne(pad_id).sum().item()  # исключаем padding
            epoch_train_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Обновляем progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}'
            })

        # Средняя потеря за эпоху
        avg_train_loss = epoch_train_loss / total_tokens if total_tokens > 0 else 0
        train_losses.append(avg_train_loss)

        # Валидация
        model.eval()
        eval_perplexity = evaluate(model, criterion, eval_dataloader, vocab_size, device, pad_id)
        # eval_perplexity = evaluate2(model, eval_dataloader, device, pad_id)
        eval_perplexities.append(eval_perplexity)

        # Сохраняем learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Train PPL:  {math.exp(avg_train_loss):.2f}')
        print(f'  Eval PPL:   {eval_perplexity:.2f}')
        print('-' * 50)

    # Результаты обучения
    results = {
        'train_losses': train_losses,
        'eval_perplexities': eval_perplexities,
        'learning_rates': learning_rates,
        'best_eval_ppl': min(eval_perplexities) if eval_perplexities else float('inf'),
        'final_train_loss': train_losses[-1] if train_losses else float('inf')
    }

    return results


def evaluate(model, criterion, dataloader, vocab_size: int, device: torch.device, pad_id: int) -> float:
    """Вычисление perplexity на валидационном наборе"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Прямой проход
            if hasattr(model, 'init_hidden'):
                hidden = model.init_hidden(input_ids.size(0), device)
                logits, _ = model(input_ids, hidden)
            else:
                logits = model(input_ids)

            # Подготавливаем данные для вычисления потерь
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = target_ids.reshape(-1)

            # Вычисляем потери
            loss = criterion(logits_flat, targets_flat)

            # Учитываем только действительные токены
            batch_tokens = targets_flat.ne(pad_id).sum().item()  # исключаем padding
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    # Вычисляем perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        perplexity = float('inf')

    return perplexity


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_config() -> dict:
    return {
        'WORD_THRESHOLD': 32,
        'VOCAB_SIZE': 40000,
        'batch_size': 128,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'seed': 42,
    }


def setup_environment() -> None:
    seaborn.set(palette='summer')
    nltk.download('punkt')


def load_imdb_dataset():
    return load_dataset('imdb')


def build_vocab_and_sentences(dataset, word_threshold: int, vocab_size: int):
    print("🚀 Начало препроцессинга данных...")
    print("=" * 60)
    return preprocess_data(dataset, word_threshold=word_threshold, vocab_size=vocab_size)


def build_mappings(vocab):
    word2ind = {char: i for i, char in enumerate(vocab)}
    ind2word = {i: char for char, i in word2ind.items()}
    pad_id = word2ind['<pad>']
    return word2ind, ind2word, pad_id


def split_sentences(processed_sentences, seed: int = 42):
    train_sentences, temp_sentences = train_test_split(
        processed_sentences,
        test_size=0.2,
        random_state=seed
    )
    eval_sentences, test_sentences = train_test_split(
        temp_sentences,
        test_size=0.5,
        random_state=seed
    )
    return train_sentences, eval_sentences, test_sentences


def make_datasets(train_sentences, eval_sentences, test_sentences, word2ind):
    train_dataset = WordDataset(train_sentences, word2ind)
    eval_dataset = WordDataset(eval_sentences, word2ind)
    test_dataset = WordDataset(test_sentences, word2ind)
    return train_dataset, eval_dataset, test_dataset


def make_dataloaders(train_dataset, eval_dataset, test_dataset, pad_id: int, batch_size: int):
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn_with_padding(pad_id), batch_size=batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn_with_padding(pad_id), batch_size=batch_size)
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn_with_padding(pad_id), batch_size=batch_size)
    return train_dataloader, eval_dataloader, test_dataloader


def build_model(vocab_size: int) -> nn.Module:
    return LanguageModel(vocab_size=vocab_size)


def train_and_evaluate(model: nn.Module,
                       train_dataloader: DataLoader,
                       eval_dataloader: DataLoader,
                       vocab_size: int,
                       num_epochs: int,
                       learning_rate: float,
                       pad_id: int,
                       device):
    return train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        vocab_size=vocab_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        pad_id=pad_id,
        device=device
    )


def run_experiment():
    setup_environment()
    device = get_device()
    print('device:', device)

    cfg = get_config()
    dataset = load_imdb_dataset()

    vocab, processed_sentences, word_frequencies = build_vocab_and_sentences(
        dataset, cfg['WORD_THRESHOLD'], cfg['VOCAB_SIZE']
    )

    assert '<unk>' in vocab
    assert '<bos>' in vocab
    assert '<eos>' in vocab
    assert '<pad>' in vocab
    assert len(vocab) == cfg['VOCAB_SIZE'] + 4

    vocab_size = cfg['VOCAB_SIZE'] + 4

    word2ind, ind2word, pad_id = build_mappings(vocab)

    train_sentences, eval_sentences, test_sentences = split_sentences(
        processed_sentences, seed=cfg['seed']
    )

    train_dataset, eval_dataset, test_dataset = make_datasets(
        train_sentences, eval_sentences, test_sentences, word2ind
    )

    train_dataloader, eval_dataloader, test_dataloader = make_dataloaders(
        train_dataset, eval_dataset, test_dataset, pad_id, cfg['batch_size']
    )

    model = build_model(vocab_size=vocab_size)

    results = train_and_evaluate(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        vocab_size=vocab_size,
        num_epochs=cfg['num_epochs'],
        learning_rate=cfg['learning_rate'],
        pad_id=pad_id,
        device=device
    )

    print(f"Лучшая perplexity: {results['best_eval_ppl']:.2f}")


if __name__ == '__main__':
    run_experiment()

