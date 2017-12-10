import argparse
import re
from random import shuffle
import json
import gzip

# Helper script to convert beeradvocate/ratebeer datasets to json format.

class DatasetGenerator(object):

    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        self.dataset = dataset
        self.itemPat = re.compile('^beer/beerId:')
        self.userPat = re.compile('^review/profileName:')
        self.textPat = re.compile('^review/text:')
        self.ratingPat = re.compile('^review/overall:')
        self.timePat = re.compile('^review/time:')
        self.reviewSep = re.compile('^$')
        self.zipped = zipped
        self.encoding = encoding
        self.f = None
        self.rb = False

    def split_getLast(self,text):
        split = text.split(" ", 1)
        if len(split) < 2:
            return None
        else:
            return split[1].strip()

    def open_reset_file(self):
        if self.zipped:
            self.f = gzip.open(self.dataset, "r")
        else:
            self.f = open(self.dataset, "r",encoding=self.encoding)
        
        for x in self:
            if len(x[3].split("/")) == 2:
                print("Detected ratebeer corpus")
                self.rb = True 
            else:
                print("Detected beeradvocate corpus")
                self.rb = False
            break

        self.f.seek(0)

    def rb_rating(self,val):
        #putting rating on 0-5 scale
        val = val.split("/")
        return (float(val[0])/float(val[1])) * 5

  

    def __iter__(self):
        dupeCount = 0
        item = None
        user = None
        text = None
        rating = None
        times = None
        i = 0

        if self.f is None:
            self.open_reset_file()

        f = self.f
        

        for line in f:
            if self.zipped:
                line = line.decode(self.encoding,errors="ignore")

            if(self.itemPat.search(line)):
                val = self.split_getLast(line)

                if val is None:
                    continue
                else:
                    item = val

            if(self.userPat.search(line)):
                val = self.split_getLast(line)

                if val is None:
                    continue
                else:
                    user = val

            if(self.textPat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    text = val

            if(self.ratingPat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    if self.rb:
                        val = self.rb_rating(val)
                    rating = val

            if(self.timePat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    times = val

            if(self.reviewSep.search(line)):
                if item is not None and user is not None and text is not None and rating is not None and times is not None:
                
                    yield((item, user, text, rating, times))

                    item = user = text = rating = times = None
                    i += 1
                    if i % 10000 == 0:
                        print("found {} reviews, {} exact duplicates".format(i, dupeCount))
        print("Found {} reviews, {} exact duplicates".format(i, dupeCount))



def run(args):

    data_iterator = DatasetGenerator(args.data,zipped=args.zipped,encoding=args.encoding)

    with gzip.open(args.output+'.gz', 'wb') as out:
    
        for i,u,rev,rat,ts in data_iterator:
            a = json.dumps({"reviewerID":u,"asin":i,"reviewText":rev,"overall":rat,"unixReviewTime":ts,"summary":rev[:32]})
            out.write((a+"\n").encode("utf-8"))
       


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("output",type=str)
parser.add_argument("--encoding",default="utf-8", type=str)
parser.add_argument('--gz', dest='zipped', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    run(args)