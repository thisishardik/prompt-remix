import random
import nltk
from nltk.translate.meteor_score import meteor_score
from Document import Document
import math
import copy
import spacy
nlp = spacy.load('en_core_web_sm')

def getPOStag(word):
    parsed = nlp(word)
    for token in parsed:
        return token.pos_


class Mutant_X:
    def __init__(self, generation_l, top_K, crossover_c, iterations_M, sentiment_based_neighbours, alpha, beta, replacements_Z, replLimit):
        self.generation = generation_l
        self.topK = top_K
        self.crossover = crossover_c
        self.iterations = iterations_M
        self.neighboursDictionary = sentiment_based_neighbours
        self.replacements = replacements_Z
        self.replacementsLimit = replLimit
        self.allowedPOStags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'PRON', 'SCONJ', 'VERB']
        self.alpha = alpha
        self.beta = beta

    def calculateFitness(self, originalDocument, changedDocument):

        if len(originalDocument.documentText.split()) < 3000:
            m_score = meteor_score([originalDocument.documentText.split()], changedDocument.documentText.split())
        else:
            m_score = meteor_score([originalDocument.documentText.split()[:3000]], changedDocument.documentText.split()[:3000])

        changedDocumentAuthor = changedDocument.documentAuthor
        probability = changedDocument.documentAuthorProbabilites[changedDocumentAuthor]
        totalReplacements = len(changedDocument.wordReplacementsDict)
        
        fitness_score = (self.alpha * probability) + (self.beta * (1 - m_score))
        fitness_score = 1 / fitness_score

        changedDocument.meteorScore = m_score
        changedDocument.fitnessScore = fitness_score
        changedDocument.classificationProbability = probability
        changedDocument.totalReplacements = totalReplacements

        # return meteor_score, fitness_score, probability, totalReplacements, changedDocumentAuthor

    def single_point_crossover(self, parentDocument1, parentDocument2):

        if not parentDocument1.documentTrailingSpacesWords or not parentDocument2.documentTrailingSpacesWords:
            return parentDocument1, parentDocument2

        randomPosition = random.randint(0, len(parentDocument1.documentTrailingSpacesWords.keys()) - 1)

        crossedOverDocumentText1 = ''
        for i in range(randomPosition - 1):
            (reqWord, _) = parentDocument1.documentTrailingSpacesWords[i]
            crossedOverDocumentText1 += reqWord
        for i in range(randomPosition, len(parentDocument2.documentTrailingSpacesWords)):
            (reqWord, _) = parentDocument2.documentTrailingSpacesWords[i]
            crossedOverDocumentText1 += reqWord

        crossedOverDocumentText2 = ''
        for i in range(randomPosition - 1):
            (reqWord, _) = parentDocument2.documentTrailingSpacesWords[i]
            crossedOverDocumentText2 += reqWord
        for i in range(randomPosition, len(parentDocument1.documentTrailingSpacesWords)):
            (reqWord, _) = parentDocument1.documentTrailingSpacesWords[i]
            crossedOverDocumentText2 += reqWord

        documentDictionary1= parentDocument1.wordReplacementsDict
        documentDictionary1 = {key: value for key, value in documentDictionary1.items() if key < randomPosition}
        documentDictionary2 = parentDocument2.wordReplacementsDict
        documentDictionary2 = {key: value for key, value in documentDictionary2.items() if key >= randomPosition}
        crossedOverDocumentDictionary1 = {**documentDictionary1, **documentDictionary2}

        documentDictionary2 = parentDocument2.wordReplacementsDict
        documentDictionary2 = {key: value for key, value in documentDictionary2.items() if key < randomPosition}
        documentDictionary1 = parentDocument1.wordReplacementsDict
        documentDictionary1 = {key: value for key, value in documentDictionary1.items() if key >= randomPosition}
        crossedOverDocumentDictionary2 = {**documentDictionary1, **documentDictionary2}

        crossedOverDocument1 = Document(crossedOverDocumentText1)
        crossedOverDocument1.wordReplacementsDict.update(crossedOverDocumentDictionary1)

        crossedOverDocument2 = Document(crossedOverDocumentText2)
        crossedOverDocument2.wordReplacementsDict.update(crossedOverDocumentDictionary2)

        return crossedOverDocument1, crossedOverDocument2

    def makeReplacement(self, currentDocumentOrig):

        currentDocument = copy.deepcopy(currentDocumentOrig)

        modifiedDocumentText = currentDocument.documentTrailingSpacesWords
        modifiedDocumentWordReplacementsDict = currentDocument.wordReplacementsDict
        modifiedDocumentAvailableReplacements = currentDocument.availableReplacements

        replacementsCount = math.floor(self.replacements * len(currentDocument.documentTrailingSpacesWords))

        i = 0
        while i < replacementsCount:
            try:
                randomPosition = random.choice(modifiedDocumentAvailableReplacements)
            except:
                return currentDocumentOrig

            try:
                (randomWord,posTag) = currentDocumentOrig.documentTrailingSpacesWords[randomPosition]
            except Exception as e:
                modifiedDocumentAvailableReplacements.remove(randomPosition)
                continue

            if posTag not in self.allowedPOStags:
                modifiedDocumentAvailableReplacements.remove(randomPosition)
                continue

            if randomWord[-1] == ' ':
                tempRandomWord = randomWord[:-1]
                try:
                    replacement = random.choice(self.neighboursDictionary[tempRandomWord])
                except:
                    modifiedDocumentAvailableReplacements.remove(randomPosition)
                    continue

                replacementPOSTag = getPOStag(replacement)
                replacement = replacement + ' '

            else:
                tempRandomWord = randomWord
                try:
                    replacement = random.choice(self.neighboursDictionary[tempRandomWord])
                except:
                    modifiedDocumentAvailableReplacements.remove(randomPosition)
                    continue

                replacementPOSTag = getPOStag(replacement)

            modifiedDocumentText[randomPosition] = (replacement, replacementPOSTag)

            try:
                modifiedDocumentWordReplacementsDict[randomPosition].append((randomWord,replacement))
            except KeyError:
                modifiedDocumentWordReplacementsDict[randomPosition] = [(randomWord, replacement)]

            i+=1

        updatedText = ''
        for i in range(len(modifiedDocumentText)):
            (reqWord,_) = modifiedDocumentText[i]
            updatedText+=reqWord

        modifiedDocument = Document(updatedText)
        modifiedDocument.wordReplacementsDict.update(modifiedDocumentWordReplacementsDict)
        modifiedDocument.availableReplacements = modifiedDocumentAvailableReplacements
        return modifiedDocument