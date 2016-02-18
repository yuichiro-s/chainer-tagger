import sys
import os


def main(args):
    for filename in sorted(os.listdir(args.section)):
        if filename.endswith('.onf'):
            path = os.path.join(args.section, filename)
            with open(path) as f:
                # one paragraph
                lst = []
                flag = False
                sentences = []
                for line in f:
                    if flag:
                        lst.append(line.rstrip())
                        if line.startswith('Treebanked sentence:'):
                            sentence = ' '.join(map(lambda l: l.lstrip(), lst[1:-2]))
                            sentences.append(sentence)
                            flag = False
                            lst = []
                    else:
                        if line.startswith('Plain sentence:'):
                            flag = True

                boundaries = set()
                idx = 0
                for sen in sentences:
                    idx += len(sen)
                    boundaries.add(idx-1)
                    idx += 1

                for i, c in enumerate(' '.join(sentences)):
                    print '\t'.join([str(i+1), c, '1' if i in boundaries else '0'])
            print


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generates PDF of LaTeX math expressions')

    parser.add_argument('section', help='WSJ section directory')

    main(parser.parse_args())
