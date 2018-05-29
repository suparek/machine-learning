import re
# import pymongo
from tqdm import tqdm
# import hashlib
import os
from collections import defaultdict
import numpy as np
import multiprocessing
import time

'''
完整的算法步骤如下：

第一步，统计：选取某个固定的nn，统计2grams、3grams、…、ngrams，计算它们的内部凝固度，只保留高于某个阈值的片段，构成一个集合GG；这一步，可以为2grams、3grams、…、ngrams设置不同的阈值，不一定要相同，因为字数越大，一般来说统计就越不充分，越有可能偏高，所以字数越大，阈值要越高；

第二步，切分：用上述grams对语料进行切分（粗糙的分词），并统计频率。切分的规则是，只有一个片段出现在前一步得到的集合GG中，这个片段就不切分，比如“各项目”，只要“各项”和“项目”都在GG中，这时候就算“各项目”不在GG中，那么“各项目”还是不切分，保留下来；

第三步，回溯：经过第二步，“各项目”会被切出来（因为第二步保证宁放过，不切错）。回溯就是检查，如果它是一个小于等于nn字的词，那么检测它在不在GG中，不在就出局；如果它是一个大于nn字的词，那个检测它每个nn字片段是不是在GG中，只要有一个片段不在，就出局。还是以“各项目”为例，回溯就是看看，“各项目”在不在3gram中，不在的话，就得出局。

每一步的补充说明：

1、较高的凝固度，但综合考虑多字，是为了更准，比如两字的“共和”不会出现在高凝固度集合中，所以会切开（比如“我一共和三个人去玩”，“共和”就切开了），但三字“共和国”出现在高凝固度集合中，所以“中华人民共和国”的“共和”不会切开；

2、第二步就是根据第一步筛选出来的集合，对句子进行切分（你可以理解为粗糙的分词），然后把“粗糙的分词结果”做统计，注意现在是统计分词结果，跟第一步的凝固度集合筛选没有交集，我们认为虽然这样的分词比较粗糙，但高频的部分还是靠谱的，所以筛选出高频部分；

3、第三步，例如因为“各项”和“项目”都出现高凝固度的片段中，所以第二步我们也不会把“各项目”切开，但我们不希望“各项目”成词，因为“各”跟“项目”的凝固度不高（“各”跟“项”的凝固度高，不代表“各”跟“项目”的凝固度高），所以通过回溯，把“各项目”移除（只需要看一下“各项目”在不在原来统计的高凝固度集合中即可，所以这步计算量是很小的）

'''

def walk_path(root_dir):
    all_text = ""
    for parent,dirnames,filenames in os.walk(root_dir):
        for file_name in filenames:
            file_path = parent+"/"+file_name
            result = []
 
            if os.path.exists(file_path):
                #print(file_path)
                with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
                    # yield f.read().strip()
                    # for t in re.split('[^\u4e00-\u9fa50-9a-zA-Z]+', f.read().strip()):
                    #     if t:
                    #         yield t
                    for text in tqdm(f.readlines()):
                        node = re.split('[^\u4e00-\u9fa50-9a-zA-Z]+', text.strip())
                        result.append(node)
                    yield result


def word_input():
    text = "华尔街向来都是资本主义至上。但理查德·克雷布认为,华尔街还可以是一个友好合作的地方。他在旧金山创立的对冲基金Numerai依靠人工智能算法来处理所有的交易。但这位现年29岁的南非数学家并不是依靠一己之力开发出这些算法的。相反,他的基金从成千上万名匿名数据科学家那里众包这些算法,那些科学家通过打造最成功的交易模型来争夺比特币奖励。而那还不是最奇怪的部分。"
    for t in re.split('[^\u4e00-\u9fa50-9a-zA-Z]+', text.strip()):
        if t:
            yield t

#第一步实现多进程
def statis_ngrams_multi(windowsize,min_count,root_dir):
    '''根据窗口大小限制先进行完全模式切词,多进程入口'''
    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count)

    ngrams = defaultdict(int)#最后返回的结果

    res = {}
    index = 0

    for textset in walk_path(root_dir):
        index += 1
        res[index] = pool.apply_async(statis_ngrams_main, (textset,windowsize,min_count,)) #将所有文本分配到各个进程,返回的是一个个列表,列表里是统计的字典

    pool.close()
    pool.join()

    for i in res:
        real_res = res[i].get()
        for ng_child in real_res:
            ngrams = merge_dict(ngrams,ng_child)

    ngrams = {i:j for i,j in ngrams.items() if j >= min_count}
    return ngrams

def merge_dict(x,y):
    '''合并字典，相同key的value相加'''
    xkeys = x.keys()
    for k, v in y.items():
        if k in xkeys:
            x[k] += v
        else:
            x[k] = v
    return x

def statis_ngrams_main(textset,windowsize,min_count):
    '''根据窗口大小限制先进行完全模式切词,主函数'''
    res = []
    for textset_sm in textset:
        for text in textset_sm:
            node = statis_ngrams(text,windowsize,min_count)
            res.append(node)
    return res

def statis_ngrams(text,windowsize,min_count):
    '''根据窗口大小限制先进行完全模式切词,功能实现'''
    ngrams = defaultdict(int)
    # for text in walk_path(root_dir):
        # print(len(text))
    for i in range(len(text)):
        for j in range(1, windowsize+1):
            if i+j <= len(text):
                ngrams[text[i:i+j]] += 1

    # ngrams = {i:j for i,j in ngrams.items() if j >= min_count}
    return ngrams

#第一步

def alone_count(ngrams):
    '''据苏建林说，1gram 2gram 3gram的总数几乎都相等，所以取1gram的总数就好'''
    total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])
    return total

# total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])



def is_keep(phrase, ngrams, total, min_proba):
    '''计算内部凝固度是否达到了阀值'''
    if len(phrase) >= 2:
        score = min([total*ngrams[phrase]/(ngrams[phrase[:i+1]]*ngrams[phrase[i+1:]]) for i in range(len(phrase)-1)])
        if score > min_proba[len(phrase)]:
            return True
    else:
        return False

def high_ngrams(ngrams,total,min_proba):
    '''筛选合格的高频词组'''
    ngrams_ = set(i for i,j in ngrams.items() if is_keep(i, ngrams, total, min_proba))
    return ngrams_


#第三步多进程
def cut_text_multi(windowsize,min_count,ngrams_,root_dir):
    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count)

    words = defaultdict(int)
    res = {}

    index = 0
    for textset in walk_path(root_dir):
        index += 1
        res[index] = pool.apply_async(cut_text_main,(textset,windowsize,min_count,ngrams_,))

    pool.close()
    pool.join()

    for i in res:
        real_res = res[i].get()
        # print(real_res)
        words = merge_dict(words,real_res)

    words = {i:j for i,j in words.items() if j >= min_count}
    return words

def cut_text_main(textset,windowsize,min_count,ngrams_):
    words = defaultdict(int)
    for textset_sm in textset:
        for text in textset_sm:
            if text:
                for i in cut(text,windowsize,ngrams_):
                    words[i] += 1

    # words = {i:j for i,j in words.items() if j >= min_count}
    return words

# def cut_text(windowsize,min_count,ngrams_,root_dir):
#     '''用得到的grams对语料进行切分（粗糙的分词），并统计频率'''
#     words = defaultdict(int)
#     for t in walk_path(root_dir):
#         for i in cut(t,windowsize,ngrams_):
#             words[i] += 1
    
#     words = {i:j for i,j in words.items() if j >= min_count}
#     return words

def cut(s,windowsize,ngrams_):
    '''切分并统计频率'''
    r = np.array([0]*(len(s)-1))
    for i in range(len(s)-1):
        for j in range(2, windowsize+1):
            if s[i:i+j] in ngrams_:
                r[i:i+j-1] += 1
    w = [s[0]]
    for i in range(1, len(s)):
        if r[i-1] > 0:
            w[-1] += s[i]
        else:
            w.append(s[i])
    return w

#第三步


def is_real(s,windowsize,ngrams_):
    '''回溯操作'''
    if len(s) >= 3:
        for i in range(3, windowsize+1):
            for j in range(len(s)-i+1):
                if s[j:j+i] not in ngrams_:
                    return False
        return True
    else:
        return True

def third(words,windowsize,ngrams_high):
    w = {i:j for i,j in words.items() if is_real(i,windowsize,ngrams_high) and len(i) > 1}
    return w

def write_words_tofile(words_fn,worddict):
    '''将结果写入文件'''
    with open(words_fn,'w+',encoding='utf-8') as file:
        writewords = [ i + ',' + str(j) + '\n' for i,j in worddict.items()]
        writewords[-1] = writewords[-1].strip()
        file.writelines(writewords)
    return True

def newword_main():
    windowsize = 6 #windowsize
    min_count = 20 #出现频次
    MIN_PROBA = {2:5, 3:25, 4:125, 5:625, 6:3125} #对于各个窗口所设的凝固度阀值

    rootdir = "测试语料"

    start = time.time()

    ngrams = statis_ngrams_multi(windowsize,min_count,rootdir)

    time1 = time.time()
    print("第一步计算完毕,耗时" + str(time1 - start) + '秒')



    total = alone_count(ngrams)
    ngrams_high = high_ngrams(ngrams,total,MIN_PROBA)

    time2 = time.time()
    print("第二步计算完毕,耗时" + str(time2 - time1) + '秒')

    words = cut_text_multi(windowsize,min_count,ngrams_high,rootdir)
    result = third(words,windowsize,ngrams_high)

    time3 = time.time()
    print("第三步计算完毕,耗时" + str(time3 - time2) + '秒')

    write_words_tofile("测试结果/结果.txt",result)

    print("多进程一共耗时" + str(time3 - start) + '秒')

def main():
    newword_main()

if __name__ == '__main__':
    main()