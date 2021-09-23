#pragma once

#ifndef CORE_UTIL_TEXTWRITER_H_
#define CORE_UTIL_TEXTWRITER_H_

namespace ml {

class TextWriter
{
public:
    TextWriter() {}
    TextWriter(const ColorImageR8G8B8A8 &bitmapFont)
    {
        init(bitmapFont);
    }

    void init(const ColorImageR8G8B8A8 &bitmapFont)
    {
        ASCIICharacters.resize(128);
        int curCharacter = 32;
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
            {
                if (curCharacter >= ASCIICharacters.size()) break;
                ColorImageR8G8B8A8 &img = ASCIICharacters[curCharacter++];
                img = bitmapFont.getSubregion(bbox2i(vec2i(x * 14, y * 16), vec2i((x + 1) * 14, (y + 1) * 16)));
            }
    }

    void writeNumber(ColorImageR8G8B8A8 &image, float value, int maxCharacters, const vec2i &coord)
    {
        std::string s = std::to_string(value);
        if (s.length() > maxCharacters)
            s.resize(maxCharacters);
        writeText(image, s, coord);
    }

    void writeText(ColorImageR8G8B8A8 &image, const std::string &text, const vec2i &coord)
    {
        int xOffset = 0;
        for (int c = 0; c < text.size(); c++)
        {
            int cValue = (int)text[c];
            if (cValue < 0 || cValue >= ASCIICharacters.size())
                continue;
            const ColorImageR8G8B8A8 &character = ASCIICharacters[cValue];
            image.copyIntoImage(character, coord.x + xOffset, coord.y, false);	//copies irrespective whether dimensions match up (claps accordingly)
            xOffset += character.getWidth();
        }
    }

private:
    std::vector<ColorImageR8G8B8A8> ASCIICharacters;
};

} // namespace ml


#endif // CORE_UTIL_TEXTWRITER_H_