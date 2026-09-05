#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

using namespace llvm;
namespace fs = std::filesystem;


// ============================================================
// Kyber paths
// ============================================================

static const fs::path KYBER_ROOT =
    R"(C:\Users\Admin\kyber)";

static const fs::path OBJECT_DIRECTORY =
    KYBER_ROOT / ".obj";

static const fs::path KYBER_OBJECT =
    OBJECT_DIRECTORY / "kyber.obj";

// Default location of the Ninja interpreter (NINJA--COMPILER.py)
// used by #ninja blocks. Override at runtime with the
// KYBER_NINJA_INTERPRETER environment variable if it lives
// somewhere else.
static const fs::path NINJA_INTERPRETER_DEFAULT =
    KYBER_ROOT / "ninja" / "NINJA--COMPILER.py";

// ============================================================
// Token system
// ============================================================

enum class TokenKind {
    Identifier,
    Number,
    String,
    Keyword,
    Operator,

    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenBrace,
    CloseBrace,

    Comma,
    Colon,
    Arrow,

    Newline,
    End
};

struct Token {
    TokenKind kind;
    std::string text;
    size_t line;
    size_t column;
};

// ============================================================
// Lexer
// ============================================================

class KyberLexer {

    std::string source;

    size_t pos = 0;
    size_t line = 1;
    size_t column = 1;

    bool eof() const {
        return pos >= source.size();
    }

    char peek(size_t n = 0) const {

        if (pos + n >= source.size())
            return '\0';

        return source[pos + n];
    }

    char advance() {

        if (eof())
            return '\0';

        char c = source[pos++];

        if (c == '\n') {
            ++line;
            column = 1;
        }
        else {
            ++column;
        }

        return c;
    }

    void skipSpaces() {

        while (!eof()) {

            char c = peek();

            if (c == ' ' ||
                c == '\t' ||
                c == '\r') {

                advance();
                continue;
                  if (source.compare(
                        pos,
                        6,
                        "#ninja"
                    ) == 0) {

                    break;
                }
  		
            }

            // Kyber comments
            if (c == '#') {

                // #extern and #n8n are handled before lexing,
                // so if either marker somehow reaches the
                // lexer (extraction was skipped or failed),
                // stop rather than silently eating it as a
                // comment.
                if (source.compare(
                        pos,
                        7,
                        "#extern"
                    ) == 0) {

                    break;
                }

                if (source.compare(
                        pos,
                        4,
                        "#n8n"
                    ) == 0) {

                    break;
                }

                if (source.compare(
                        pos,
                        6,
                        "#ninja"
                    ) == 0) {

                    break;
                }

                while (!eof() &&
                       peek() != '\n') {

                    advance();
                }

                continue;
            }

            break;
        }
    }

    Token identifier() {

        size_t startLine = line;
        size_t startColumn = column;

        std::string text;

        while (!eof()) {

            char c = peek();

            if (std::isalnum(
                    static_cast<unsigned char>(c)
                ) ||
                c == '_') {

                text += advance();
            }
            else {
                break;
            }
        }

        static const std::unordered_set<std::string>
            keywords = {

                "fn",
                "let",
                "tensor",
                "matrix",
                "layer",
                "target",

                "cpu",
                "gpu",
                               // I/O
                "print",
                "input",

                // File handling
                "file",
                "open",
                "read",
                "write",
                "append",
                "close",

                // Error handling
                "try",
                "catch",
                "throw",
                "error",

                // C/C++ interop
                "extern",
                "cpp",
                "c",
                "include",
                "call"

                "return",
                "if",
                "else",
                "for",
                "in",

                "true",
                "false",

                "Dense",
                "ReLU",
                "Conv",
                "Sigmoid",
                "Softmax"
            };

        TokenKind kind =
            keywords.count(text)
                ? TokenKind::Keyword
                : TokenKind::Identifier;

        return {
            kind,
            text,
            startLine,
            startColumn
        };
    }

    Token number() {

        size_t startLine = line;
        size_t startColumn = column;

        std::string text;

        bool decimal = false;

        while (!eof()) {

            char c = peek();

            if (std::isdigit(
                    static_cast<unsigned char>(c)
                )) {

                text += advance();
                continue;
            }

            if (c == '.' && !decimal) {

                decimal = true;
                text += advance();
                continue;
            }

            break;
        }

        return {
            TokenKind::Number,
            text,
            startLine,
            startColumn
        };
    }

    Token stringLiteral() {

        size_t startLine = line;
        size_t startColumn = column;

        advance(); // "

        std::string text;

        while (!eof() &&
               peek() != '"') {

            char c = advance();

            if (c == '\\' && !eof()) {

                char next = advance();

                if (next == 'n')
                    text += '\n';

                else if (next == 't')
                    text += '\t';

                else
                    text += next;
            }
            else {
                text += c;
            }
        }

        if (!eof())
            advance();

        return {
            TokenKind::String,
            text,
            startLine,
            startColumn
        };
    }

public:

    explicit KyberLexer(
        const std::string& src
    )
        : source(src) {}

    std::vector<Token> tokenize() {

        std::vector<Token> tokens;

        while (!eof()) {

            skipSpaces();

            if (eof())
                break;

            char c = peek();

            size_t tokenLine = line;
            size_t tokenColumn = column;

            if (c == '\n') {

                advance();

                tokens.push_back({
                    TokenKind::Newline,
                    "\\n",
                    tokenLine,
                    tokenColumn
                });

                continue;
            }

            if (std::isalpha(
                    static_cast<unsigned char>(c)
                ) ||
                c == '_') {

                tokens.push_back(
                    identifier()
                );

                continue;
            }

            if (std::isdigit(
                    static_cast<unsigned char>(c)
                )) {

                tokens.push_back(
                    number()
                );

                continue;
            }

            if (c == '"') {

                tokens.push_back(
                    stringLiteral()
                );

                continue;
            }

            if (c == '-' &&
                peek(1) == '>') {

                advance();
                advance();

                tokens.push_back({
                    TokenKind::Arrow,
                    "->",
                    tokenLine,
                    tokenColumn
                });

                continue;
            }

            TokenKind kind;

            switch (c) {

                case '(':
                    kind = TokenKind::OpenParen;
                    break;

                case ')':
                    kind = TokenKind::CloseParen;
                    break;

                case '[':
                    kind = TokenKind::OpenBracket;
                    break;

                case ']':
                    kind = TokenKind::CloseBracket;
                    break;

                case '{':
                    kind = TokenKind::OpenBrace;
                    break;

                case '}':
                    kind = TokenKind::CloseBrace;
                    break;

                case ',':
                    kind = TokenKind::Comma;
                    break;

                case ':':
                    kind = TokenKind::Colon;
                    break;

                default:
                    kind = TokenKind::Operator;
                    break;
            }

            std::string text(1, advance());

            if (kind == TokenKind::Operator) {

                if ((text == "=" ||
                     text == "!" ||
                     text == "<" ||
                     text == ">") &&
                    peek() == '=') {

                    text += advance();
                }
            }

            tokens.push_back({
                kind,
                text,
                tokenLine,
                tokenColumn
            });
        }

        tokens.push_back({
            TokenKind::End,
            "",
            line,
            column
        });

        return tokens;
    }
};

// ============================================================
// AST
// ============================================================

struct ASTNode {

    virtual ~ASTNode() = default;

    virtual void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) = 0;
};

// ============================================================
// Tensor
// ============================================================

struct TensorNode : ASTNode {

    std::string name;
    std::vector<double> values;

    TensorNode(
        std::string n,
        std::vector<double> v
    )
        : name(std::move(n)),
          values(std::move(v)) {}

    void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) override {

        if (values.empty())
            return;

        auto* elementType =
            llvm::Type::getDoubleTy(context);

        auto* arrayType =
            llvm::ArrayType::get(
                elementType,
                values.size()
            );

        std::vector<llvm::Constant*> constants;

        for (double value : values) {

            constants.push_back(
                llvm::ConstantFP::get(
                    elementType,
                    value
                )
            );
        }

        auto* initializer =
            llvm::ConstantArray::get(
                arrayType,
                constants
            );

        new llvm::GlobalVariable(
            module,
            arrayType,
            true,
            llvm::GlobalValue::InternalLinkage,
            initializer,
            name
        );
    }
};

// ============================================================
// Matrix
// ============================================================

struct MatrixNode : ASTNode {

    std::string name;

    std::vector<std::vector<double>>
        values;

    MatrixNode(
        std::string n,
        std::vector<std::vector<double>> v
    )
        : name(std::move(n)),
          values(std::move(v)) {}

    void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) override {

        if (values.empty())
            return;

        size_t rows =
            values.size();

        size_t cols =
            values[0].size();

        if (cols == 0)
            return;

        auto* elementType =
            llvm::Type::getDoubleTy(context);

        auto* rowType =
            llvm::ArrayType::get(
                elementType,
                cols
            );

        auto* matrixType =
            llvm::ArrayType::get(
                rowType,
                rows
            );

        std::vector<llvm::Constant*>
            rowConstants;

        for (const auto& row : values) {

            if (row.size() != cols)
                return;

            std::vector<llvm::Constant*>
                elements;

            for (double value : row) {

                elements.push_back(
                    llvm::ConstantFP::get(
                        elementType,
                        value
                    )
                );
            }

            rowConstants.push_back(
                llvm::ConstantArray::get(
                    rowType,
                    elements
                )
            );
        }

        auto* initializer =
            llvm::ConstantArray::get(
                matrixType,
                rowConstants
            );

        new llvm::GlobalVariable(
            module,
            matrixType,
            true,
            llvm::GlobalValue::InternalLinkage,
            initializer,
            name
        );
    }
};

// ============================================================
// Layer
// ============================================================

struct LayerNode : ASTNode {

    std::string name;
    std::string layerType;

    std::vector<double>
        arguments;

    LayerNode(
        std::string n,
        std::string t,
        std::vector<double> a
    )
        : name(std::move(n)),
          layerType(std::move(t)),
          arguments(std::move(a)) {}

    void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) override {

        auto* structType =
            llvm::StructType::create(
                context,
                "kyber.layer." + name
            );

        std::vector<llvm::Type*>
            fields;

        for (size_t i = 0;
             i < arguments.size();
             ++i) {

            fields.push_back(
                llvm::Type::getDoubleTy(
                    context
                )
            );
        }

        if (fields.empty()) {

            fields.push_back(
                llvm::Type::getInt32Ty(
                    context
                )
            );
        }

        structType->setBody(
            fields
        );

        auto* initializer =
            llvm::Constant::getNullValue(
                structType
            );

        new llvm::GlobalVariable(
            module,
            structType,
            false,
            llvm::GlobalValue::InternalLinkage,
            initializer,
            name
        );
    }
};

// ============================================================
// Math
// ============================================================

struct MathNode : ASTNode {

    std::string name;
    std::string operation;

    std::vector<std::string>
        operands;

    MathNode(
        std::string n,
        std::string op,
        std::vector<std::string> args
    )
        : name(std::move(n)),
          operation(std::move(op)),
          operands(std::move(args)) {}

    void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) override {

        auto* mainFunction =
            module.getFunction("main");

        if (!mainFunction)
            return;

        auto* block =
            &mainFunction->getEntryBlock();

        llvm::IRBuilder<> builder(
            context
        );

        builder.SetInsertPoint(
            block,
            block->end()
        );

        auto* doubleType =
            llvm::Type::getDoubleTy(
                context
            );

        if (operation == "add" &&
            operands.size() == 2) {

            auto* a =
                llvm::ConstantFP::get(
                    doubleType,
                    0.0
                );

            auto* b =
                llvm::ConstantFP::get(
                    doubleType,
                    0.0
                );

            builder.CreateFAdd(
                a,
                b,
                name
            );
        }
    }
};

// ============================================================
// Program
// ============================================================

struct ProgramNode : ASTNode {

    std::vector<
        std::unique_ptr<ASTNode>
    > items;

    void codegen(
        llvm::Module& module,
        llvm::LLVMContext& context
    ) override {

        llvm::IRBuilder<> builder(
            context
        );

        auto* functionType =
            llvm::FunctionType::get(
                builder.getInt32Ty(),
                false
            );

        auto* mainFunction =
            llvm::Function::Create(
                functionType,
                llvm::Function::ExternalLinkage,
                "main",
                module
            );

        auto* block =
            llvm::BasicBlock::Create(
                context,
                "entry",
                mainFunction
            );

        builder.SetInsertPoint(block);

        for (auto& item : items)
            item->codegen(
                module,
                context
            );

        if (!block->getTerminator()) {

            builder.CreateRet(
                builder.getInt32(0)
            );
        }
    }
};

// ============================================================
// Parser
// ============================================================

class KyberParser {

    const std::vector<Token>& tokens;

    size_t current = 0;

    const Token& peek() const {
        return tokens[current];
    }

    bool is(TokenKind kind) const {
        return peek().kind == kind;
    }

    bool isText(
        const std::string& text
    ) const {
        return peek().text == text;
    }

    Token advance() {
        return tokens[current++];
    }

    bool match(TokenKind kind) {

        if (!is(kind))
            return false;

        advance();
        return true;
    }

    bool matchText(
        const std::string& text
    ) {

        if (!isText(text))
            return false;

        advance();
        return true;
    }

    void skipNewlines() {

        while (is(TokenKind::Newline))
            advance();
    }

    bool expect(
        TokenKind kind,
        const std::string& message
    ) {

        if (is(kind)) {
            advance();
            return true;
        }

        std::cerr
            << "Parser error at line "
            << peek().line
            << ": "
            << message
            << "\n";

        return false;
    }

    std::vector<double>
    parseNumberList() {

        std::vector<double> values;

        if (!expect(
                TokenKind::OpenBracket,
                "expected '['"
            )) {

            return values;
        }

        while (!is(TokenKind::CloseBracket) &&
               !is(TokenKind::End)) {

            if (!is(TokenKind::Number)) {

                std::cerr
                    << "Expected number at line "
                    << peek().line
                    << "\n";

                return {};
            }

            values.push_back(
                std::stod(
                    advance().text
                )
            );

            if (!match(TokenKind::Comma))
                break;
        }

        expect(
            TokenKind::CloseBracket,
            "expected ']'"
        );

        return values;
    }

    std::vector<
        std::vector<double>
    > parseMatrix() {

        std::vector<
            std::vector<double>
        > matrix;

        if (!expect(
                TokenKind::OpenBracket,
                "expected '['"
            )) {

            return matrix;
        }

        while (!is(TokenKind::CloseBracket) &&
               !is(TokenKind::End)) {

            matrix.push_back(
                parseNumberList()
            );

            if (!match(TokenKind::Comma))
                break;
        }

        expect(
            TokenKind::CloseBracket,
            "expected ']'"
        );

        return matrix;
    }

    std::vector<double>
    parseCallArguments() {

        std::vector<double> args;

        if (!expect(
                TokenKind::OpenParen,
                "expected '('"
            )) {

            return args;
        }

        while (!is(TokenKind::CloseParen) &&
               !is(TokenKind::End)) {

            if (!is(TokenKind::Number)) {

                std::cerr
                    << "Expected number at line "
                    << peek().line
                    << "\n";

                return {};
            }

            args.push_back(
                std::stod(
                    advance().text
                )
            );

            if (!match(TokenKind::Comma))
                break;
        }

        expect(
            TokenKind::CloseParen,
            "expected ')'"
        );

        return args;
    }

    std::unique_ptr<ASTNode>
    parseTensor() {

        advance();

        if (!is(TokenKind::Identifier)) {

            std::cerr
                << "Expected tensor name\n";

            return nullptr;
        }

        std::string name =
            advance().text;

        if (!matchText("=")) {

            std::cerr
                << "Expected '=' after tensor name\n";

            return nullptr;
        }

        auto values =
            parseNumberList();

        return std::make_unique<TensorNode>(
            name,
            values
        );
    }

    std::unique_ptr<ASTNode>
    parseMatrixNode() {

        advance();

        if (!is(TokenKind::Identifier)) {

            std::cerr
                << "Expected matrix name\n";

            return nullptr;
        }

        std::string name =
            advance().text;

        if (!matchText("=")) {

            std::cerr
                << "Expected '=' after matrix name\n";

            return nullptr;
        }

        auto values =
            parseMatrix();

        return std::make_unique<MatrixNode>(
            name,
            values
        );
    }

    std::unique_ptr<ASTNode>
    parseLayer() {

        advance();

        if (!is(TokenKind::Identifier)) {

            std::cerr
                << "Expected layer name\n";

            return nullptr;
        }

        std::string name =
            advance().text;

        if (!matchText("=")) {

            std::cerr
                << "Expected '=' after layer name\n";

            return nullptr;
        }

        if (!is(TokenKind::Identifier) &&
            !is(TokenKind::Keyword)) {

            std::cerr
                << "Expected layer type\n";

            return nullptr;
        }

        std::string type =
            advance().text;

        auto args =
            parseCallArguments();

        return std::make_unique<LayerNode>(
            name,
            type,
            args
        );
    }

    std::unique_ptr<ASTNode>
    parseMath() {

        if (!is(TokenKind::Identifier))
            return nullptr;

        std::string name =
            advance().text;

        if (!matchText("="))
            return nullptr;

        if (!is(TokenKind::Identifier) &&
            !is(TokenKind::Keyword)) {

            return nullptr;
        }

        std::string operation =
            advance().text;

        if (!expect(
                TokenKind::OpenParen,
                "expected '('"
            )) {

            return nullptr;
        }

        std::vector<std::string>
            operands;

        while (!is(TokenKind::CloseParen) &&
               !is(TokenKind::End)) {

            if (is(TokenKind::Identifier) ||
                is(TokenKind::Number)) {

                operands.push_back(
                    advance().text
                );
            }
            else {
                return nullptr;
            }

            if (!match(TokenKind::Comma))
                break;
        }

        if (!expect(
                TokenKind::CloseParen,
                "expected ')'"
            )) {

            return nullptr;
        }

        return std::make_unique<MathNode>(
            name,
            operation,
            operands
        );
    }

    std::unique_ptr<ASTNode>
    parseStatement() {

        if (isText("tensor"))
            return parseTensor();

        if (isText("matrix"))
            return parseMatrixNode();

        if (isText("layer"))
            return parseLayer();

        if (is(TokenKind::Identifier))
            return parseMath();

        // target cpu / target gpu
        if (isText("target")) {

            advance();

            if (is(TokenKind::Identifier) ||
                is(TokenKind::Keyword)) {

                advance();
            }

            return nullptr;
        }

        advance();

        return nullptr;
    }

public:

    explicit KyberParser(
        const std::vector<Token>& t
    )
        : tokens(t) {}

    std::unique_ptr<ProgramNode>
    parse() {

        auto program =
            std::make_unique<ProgramNode>();

        while (!is(TokenKind::End)) {

            skipNewlines();

            if (is(TokenKind::End))
                break;

            auto node =
                parseStatement();

            if (node) {

                program->items.push_back(
                    std::move(node)
                );
            }

            skipNewlines();
        }

        return program;
    }
};

// ============================================================
// Extern blocks
// ============================================================

struct ExternBlock {

    std::string language;
    std::string source;
};

// ============================================================
// n8n blocks
//
// Example:
//
// #n8n "send_alert"
// {
//     url: "https://your-n8n-host/webhook/abc123"
//     method: "POST"
//     payload: "{\"status\":\"alert\"}"
// }
//
// Each #n8n block is compiled, like an #extern block, into
// its own object file exposing a C function of the same name:
//
//     int send_alert(const char* payload);
//
// Calling that function performs an HTTP request to the
// configured n8n webhook URL, so Kyber programs (or any code
// linked against kyber.obj) can trigger n8n workflows. Passing
// NULL for payload falls back to the payload literal given in
// the block, if any.
// ============================================================

struct N8nBlock {

    std::string name;
    std::string url;
    std::string method = "POST";
    std::string payload;
};

// ============================================================
// Ninja blocks
//
// Ninja (NINJA--COMPILER.py) is a separate, standalone
// interpreter — it tokenizes, parses, and directly executes a
// .ninja program; it does not emit machine code or object
// files. That means it can't be lowered into LLVM IR and
// linked the way Kyber's own tensor/matrix/layer nodes are.
//
// What IS possible: treat a #ninja block like an #extern
// block, but instead of compiling the embedded source with
// clang, embed it as a string constant and generate a tiny C
// wrapper function that, at RUNTIME, writes that string to a
// temp .ninja file and shells out to the Ninja interpreter via
// "python3 NINJA--COMPILER.py <temp file>". That wrapper still
// compiles to a normal .obj (kyber_ninja_N.obj) that kyberlink
// can link in like any other extern symbol.
//
// Caveats worth knowing:
//   - This requires python3 and NINJA--COMPILER.py to be
//     present on the machine that RUNS the final executable,
//     not just the machine that compiles it.
//   - The Kyber program only gets the Ninja process's exit
//     code back, not structured return values, since Ninja
//     has no notion of returning a value to a caller.
//
// Example:
//
// #ninja "run_forward_pass"
// {
//     VOID MAIN() {
//         SCREEN.PRINT("hello from ninja")
//     }
// }
// ============================================================

struct NinjaBlock {

    std::string name;
    std::string source;
};

// ============================================================
// Generic helpers shared by block extraction
// ============================================================

static std::string trimWhitespace(
    const std::string& text
) {

    size_t start =
        text.find_first_not_of(" \t\r\n");

    if (start == std::string::npos)
        return "";

    size_t end =
        text.find_last_not_of(" \t\r\n");

    return text.substr(
        start,
        end - start + 1
    );
}

static std::string unquote(
    const std::string& text
) {

    if (text.size() >= 2 &&
        text.front() == '"' &&
        text.back() == '"') {

        return text.substr(
            1,
            text.size() - 2
        );
    }

    return text;
}

// Escapes a string so it can be embedded safely inside a
// double-quoted C string literal.
static std::string escapeForC(
    const std::string& text
) {

    std::string result;

    for (char c : text) {

        if (c == '"' || c == '\\')
            result += '\\';

        result += c;
    }

    return result;
}

// Escapes a string so it can be embedded as the body of a
// single double-quoted C string literal, including turning
// real newlines into the two-character escape "\n". Used for
// baking multi-line source (e.g. a #ninja block's body)
// directly into generated C code.
static std::string escapeForCStringLiteral(
    const std::string& text
) {

    std::string result;

    for (char c : text) {

        switch (c) {

            case '"':
                result += "\\\"";
                break;

            case '\\':
                result += "\\\\";
                break;

            case '\n':
                result += "\\n";
                break;

            case '\r':
                result += "\\r";
                break;

            default:
                result += c;
                break;
        }
    }

    return result;
}

// Finds the matching '}' for the '{' found at `openBrace`,
// respecting nested braces and quoted strings. Returns the
// index one past the closing '}', or std::string::npos if the
// block is unterminated.
static size_t findBlockEnd(
    const std::string& source,
    size_t openBrace
) {

    int depth = 1;

    size_t i = openBrace + 1;

    bool inString = false;
    char stringChar = '\0';

    while (i < source.size() &&
           depth > 0) {

        char c = source[i];

        if ((c == '"' || c == '\'') &&
            (i == 0 ||
             source[i - 1] != '\\')) {

            if (!inString) {

                inString = true;
                stringChar = c;
            }
            else if (stringChar == c) {

                inString = false;
            }
        }

        if (!inString) {

            if (c == '{')
                ++depth;

            else if (c == '}')
                --depth;
        }

        ++i;
    }

    if (depth != 0)
        return std::string::npos;

    return i;
}

// ============================================================
// Extract #extern blocks
//
// Example:
//
// #extern "c"
// {
//     int hello() { return 1; }
// }
//
// #extern "c++"
// {
//     int world() { return 2; }
// }
// ============================================================

static bool extractExternBlocks(
    const std::string& source,
    std::string& kyberSource,
    std::vector<ExternBlock>& blocks
) {

    kyberSource.clear();

    size_t pos = 0;

    while (pos < source.size()) {

        size_t marker =
            source.find(
                "#extern",
                pos
            );

        if (marker == std::string::npos) {

            kyberSource +=
                source.substr(pos);

            break;
        }

        kyberSource +=
            source.substr(
                pos,
                marker - pos
            );

        size_t quote1 =
            source.find(
                '"',
                marker
            );

        if (quote1 == std::string::npos)
            return false;

        size_t quote2 =
            source.find(
                '"',
                quote1 + 1
            );

        if (quote2 == std::string::npos)
            return false;

        std::string language =
            source.substr(
                quote1 + 1,
                quote2 - quote1 - 1
            );

        if (language != "c" &&
            language != "c++") {

            std::cerr
                << "Unsupported #extern language: "
                << language
                << "\n";

            return false;
        }

        size_t open =
            source.find(
                '{',
                quote2
            );

        if (open == std::string::npos)
            return false;

        size_t i =
            findBlockEnd(source, open);

        if (i == std::string::npos) {

            std::cerr
                << "Unclosed #extern block\n";

            return false;
        }

        std::string body =
            source.substr(
                open + 1,
                i - open - 2
            );

        blocks.push_back({
            language,
            body
        });

        pos = i;
    }

    return true;
}

// ============================================================
// Extract #n8n blocks
// ============================================================

static bool parseN8nBody(
    const std::string& body,
    N8nBlock& block
) {

    std::istringstream stream(body);
    std::string line;

    while (std::getline(stream, line)) {

        size_t colon =
            line.find(':');

        if (colon == std::string::npos)
            continue;

        std::string key =
            trimWhitespace(
                line.substr(0, colon)
            );

        std::string value =
            unquote(
                trimWhitespace(
                    line.substr(colon + 1)
                )
            );

        if (key == "url")
            block.url = value;

        else if (key == "method")
            block.method = value;

        else if (key == "payload")
            block.payload = value;
    }

    if (block.url.empty()) {

        std::cerr
            << "#n8n block '"
            << block.name
            << "' is missing a 'url' field\n";

        return false;
    }

    if (block.method.empty())
        block.method = "POST";

    return true;
}

static bool extractN8nBlocks(
    const std::string& source,
    std::string& kyberSource,
    std::vector<N8nBlock>& blocks
) {

    kyberSource.clear();

    size_t pos = 0;

    while (pos < source.size()) {

        size_t marker =
            source.find(
                "#n8n",
                pos
            );

        if (marker == std::string::npos) {

            kyberSource +=
                source.substr(pos);

            break;
        }

        kyberSource +=
            source.substr(
                pos,
                marker - pos
            );

        size_t quote1 =
            source.find(
                '"',
                marker
            );

        if (quote1 == std::string::npos)
            return false;

        size_t quote2 =
            source.find(
                '"',
                quote1 + 1
            );

        if (quote2 == std::string::npos)
            return false;

        std::string name =
            source.substr(
                quote1 + 1,
                quote2 - quote1 - 1
            );

        size_t open =
            source.find(
                '{',
                quote2
            );

        if (open == std::string::npos)
            return false;

        size_t i =
            findBlockEnd(source, open);

        if (i == std::string::npos) {

            std::cerr
                << "Unclosed #n8n block\n";

            return false;
        }

        std::string body =
            source.substr(
                open + 1,
                i - open - 2
            );

        N8nBlock block;
        block.name = name;

        if (!parseN8nBody(body, block))
            return false;

        blocks.push_back(block);

        pos = i;
    }

    return true;
}

// ============================================================
// Extract #ninja blocks
//
// Unlike #n8n, the block body is kept verbatim as raw Ninja
// source rather than parsed into key/value fields — same shape
// as an #extern block.
// ============================================================

static bool extractNinjaBlocks(
    const std::string& source,
    std::string& kyberSource,
    std::vector<NinjaBlock>& blocks
) {

    kyberSource.clear();

    size_t pos = 0;

    while (pos < source.size()) {

        size_t marker =
            source.find(
                "#ninja",
                pos
            );

        if (marker == std::string::npos) {

            kyberSource +=
                source.substr(pos);

            break;
        }

        kyberSource +=
            source.substr(
                pos,
                marker - pos
            );

        size_t quote1 =
            source.find(
                '"',
                marker
            );

        if (quote1 == std::string::npos)
            return false;

        size_t quote2 =
            source.find(
                '"',
                quote1 + 1
            );

        if (quote2 == std::string::npos)
            return false;

        std::string name =
            source.substr(
                quote1 + 1,
                quote2 - quote1 - 1
            );

        size_t open =
            source.find(
                '{',
                quote2
            );

        if (open == std::string::npos)
            return false;

        size_t i =
            findBlockEnd(source, open);

        if (i == std::string::npos) {

            std::cerr
                << "Unclosed #ninja block\n";

            return false;
        }

        std::string body =
            source.substr(
                open + 1,
                i - open - 2
            );

        blocks.push_back({
            name,
            body
        });

        pos = i;
    }

    return true;
}

// ============================================================
// Generate the C source for a compiled #n8n block
// ============================================================

static std::string generateN8nRuntimeSource(
    const N8nBlock& block
) {

    std::ostringstream out;

    out
        << "#include <stddef.h>\n"
        << "#include <curl/curl.h>\n\n"

        << "// Auto-generated by Kyber for #n8n \""
        << block.name
        << "\"\n"

        << "int "
        << block.name
        << "(const char* payload) {\n\n"

        << "    CURL* curl = curl_easy_init();\n"
        << "    if (!curl) return -1;\n\n"

        << "    const char* effectivePayload = payload;\n"
        << "    if (!effectivePayload) {\n"
        << "        effectivePayload = "
        << (
               block.payload.empty()
                   ? "NULL"
                   : ("\"" + escapeForC(block.payload) + "\"")
           )
        << ";\n"
        << "    }\n\n"

        << "    struct curl_slist* headers = NULL;\n"
        << "    headers = curl_slist_append(\n"
        << "        headers,\n"
        << "        \"Content-Type: application/json\"\n"
        << "    );\n\n"

        << "    curl_easy_setopt(curl, CURLOPT_URL, \""
        << escapeForC(block.url)
        << "\");\n"

        << "    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, \""
        << escapeForC(block.method)
        << "\");\n"

        << "    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n\n"

        << "    if (effectivePayload) {\n"
        << "        curl_easy_setopt(\n"
        << "            curl,\n"
        << "            CURLOPT_POSTFIELDS,\n"
        << "            effectivePayload\n"
        << "        );\n"
        << "    }\n\n"

        << "    CURLcode result = curl_easy_perform(curl);\n\n"

        << "    curl_slist_free_all(headers);\n"
        << "    curl_easy_cleanup(curl);\n\n"

        << "    return result == CURLE_OK ? 0 : -1;\n"
        << "}\n";

    return out.str();
}

// ============================================================
// Generate the C source for a compiled #ninja block
// ============================================================

static std::string generateNinjaRuntimeSource(
    const NinjaBlock& block
) {

    std::ostringstream out;

    out
        << "#include <stdio.h>\n"
        << "#include <stdlib.h>\n"
        << "#include <string.h>\n\n"

        << "#ifdef _WIN32\n"
        << "#include <process.h>\n"
        << "#define KYBER_GETPID _getpid\n"
        << "#else\n"
        << "#include <unistd.h>\n"
        << "#define KYBER_GETPID getpid\n"
        << "#endif\n\n"

        << "// Auto-generated by Kyber for #ninja \""
        << block.name
        << "\"\n"
        << "// Embeds the Ninja source below and runs it through\n"
        << "// the Ninja interpreter (NINJA--COMPILER.py) via\n"
        << "// python3 at runtime. Requires python3 and the\n"
        << "// interpreter to be available on the machine that\n"
        << "// runs this program (see KYBER_NINJA_INTERPRETER).\n\n"

        << "static const char* KYBER_NINJA_SOURCE_"
        << block.name
        << " =\n    \""
        << escapeForCStringLiteral(block.source)
        << "\";\n\n"

        << "int "
        << block.name
        << "(void) {\n\n"

        << "    const char* interpreter =\n"
        << "        getenv(\"KYBER_NINJA_INTERPRETER\");\n\n"
        << "    if (!interpreter || !interpreter[0]) {\n"
        << "        interpreter = \""
        << escapeForCStringLiteral(
               NINJA_INTERPRETER_DEFAULT.string()
           )
        << "\";\n"
        << "    }\n\n"

        << "    const char* tempDir = getenv(\"TEMP\");\n"
        << "    if (!tempDir || !tempDir[0]) {\n"
        << "        tempDir = getenv(\"TMPDIR\");\n"
        << "    }\n"
        << "    if (!tempDir || !tempDir[0]) {\n"
        << "        tempDir = \"/tmp\";\n"
        << "    }\n\n"

        << "    char tempPath[1024];\n"
        << "    snprintf(\n"
        << "        tempPath,\n"
        << "        sizeof(tempPath),\n"
        << "        \"%s/kyber_ninja_"
        << block.name
        << "_%d.ninja\",\n"
        << "        tempDir,\n"
        << "        (int)KYBER_GETPID()\n"
        << "    );\n\n"

        << "    FILE* f = fopen(tempPath, \"w\");\n"
        << "    if (!f) return -1;\n"
        << "    fputs(KYBER_NINJA_SOURCE_"
        << block.name
        << ", f);\n"
        << "    fclose(f);\n\n"

        << "    char command[2048];\n"
        << "    snprintf(\n"
        << "        command,\n"
        << "        sizeof(command),\n"
        << "        \"python3 \\\"%s\\\" \\\"%s\\\"\",\n"
        << "        interpreter,\n"
        << "        tempPath\n"
        << "    );\n\n"

        << "    int result = system(command);\n\n"

        << "    remove(tempPath);\n\n"

        << "    return result;\n"
        << "}\n";

    return out.str();
}

// ============================================================
// File utilities
// ============================================================

static bool writeFile(
    const fs::path& path,
    const std::string& data
) {

    std::ofstream out(
        path,
        std::ios::binary
    );

    if (!out)
        return false;

    out << data;

    return true;
}

static std::string readFile(
    const fs::path& path
) {

    std::ifstream in(
        path,
        std::ios::binary
    );

    if (!in)
        return {};

    std::stringstream ss;

    ss << in.rdbuf();

    return ss.str();
}

// ============================================================
// Shell argument quoting
//
// Windows-safe enough for normal paths.
// ============================================================

static std::string quoteArg(
    const std::string& value
) {

    std::string result = "\"";

    for (char c : value) {

        if (c == '"')
            result += "\\\"";
        else
            result += c;
    }

    result += "\"";

    return result;
}

// ============================================================
// Compile a C/C++ source file with Clang into an object file.
//
// Shared by both #extern and #n8n compilation, since both are
// "write a temp source file, invoke clang -c, keep the .obj"
// pipelines.
// ============================================================

static bool compileSourceToObject(
    const std::string& sourceText,
    const std::string& extension,
    const std::string& compiler,
    const fs::path& objectPath,
    const std::string& tempFileTag,
    size_t index
) {

    fs::path tempDirectory =
        fs::temp_directory_path();

    fs::path tempSource =
        tempDirectory /
        (
            "kyber_" +
            tempFileTag +
            "_" +
            std::to_string(index) +
            extension
        );

    if (!writeFile(
            tempSource,
            sourceText
        )) {

        std::cerr
            << "Could not create temporary source:\n"
            << tempSource
            << "\n";

        return false;
    }

    std::error_code ec;

    fs::create_directories(
        OBJECT_DIRECTORY,
        ec
    );

    if (ec) {

        std::cerr
            << "Could not create Kyber .obj directory: "
            << ec.message()
            << "\n";

        fs::remove(tempSource, ec);

        return false;
    }

    std::string command =
        quoteArg(compiler) +
        " -c -O2 " +
        quoteArg(
            tempSource.string()
        ) +
        " -o " +
        quoteArg(
            objectPath.string()
        );

    std::cout
        << "\n[Kyber Clang]\n"
        << "Compiler: "
        << compiler
        << "\n"
        << "Output: "
        << objectPath
        << "\n";

    int result =
        std::system(
            command.c_str()
        );

    std::error_code removeError;

    fs::remove(
        tempSource,
        removeError
    );

    if (result != 0) {

        std::cerr
            << "Clang failed to compile "
            << tempFileTag
            << " block.\n";

        return false;
    }

    if (!fs::exists(objectPath)) {

        std::cerr
            << "Clang returned success, "
               "but object file was not created:\n"
            << objectPath
            << "\n";

        return false;
    }

    std::cout
        << "Generated: "
        << objectPath
        << "\n";

    return true;
}

// ============================================================
// Compile C/C++ extern block with Clang
// ============================================================

static bool compileExternBlock(
    const ExternBlock& block,
    const fs::path& objectPath,
    size_t index
) {

    if (block.language != "c" &&
        block.language != "c++") {

        std::cerr
            << "Unsupported extern language: "
            << block.language
            << "\n";

        return false;
    }

    std::string extension =
        block.language == "c"
            ? ".c"
            : ".cpp";

    std::string compiler =
        block.language == "c"
            ? "clang"
            : "clang++";

    return compileSourceToObject(
        block.source,
        extension,
        compiler,
        objectPath,
        "extern",
        index
    );
}

// ============================================================
// Compile a #n8n block into its own object file.
//
// The generated function calls into libcurl, so the FINAL LINK
// step (kyberlink.exe) needs "-lcurl" alongside the usual
// NForce libraries whenever any #n8n blocks were compiled.
// ============================================================

static bool compileN8nBlock(
    const N8nBlock& block,
    const fs::path& objectPath,
    size_t index
) {

    std::string source =
        generateN8nRuntimeSource(block);

    return compileSourceToObject(
        source,
        ".c",
        "clang",
        objectPath,
        "n8n",
        index
    );
}

// ============================================================
// Compile a #ninja block into its own object file. The
// generated wrapper shells out to python3 at runtime, so
// nothing Ninja-specific is needed at COMPILE time beyond
// clang itself.
// ============================================================

static bool compileNinjaBlock(
    const NinjaBlock& block,
    const fs::path& objectPath,
    size_t index
) {

    std::string source =
        generateNinjaRuntimeSource(block);

    return compileSourceToObject(
        source,
        ".c",
        "clang",
        objectPath,
        "ninja",
        index
    );
}

// ============================================================
// LLVM lowering
// ============================================================

static std::unique_ptr<Module>
lowerToLLVM(
    ASTNode* ast,
    LLVMContext& context,
    const std::string& moduleName
) {

    auto module =
        std::make_unique<Module>(
            moduleName,
            context
        );

    module->setTargetTriple(
        sys::getDefaultTargetTriple()
    );

    if (ast) {

        ast->codegen(
            *module,
            context
        );
    }

    return module;
}

// ============================================================
// LLVM object generation
// ============================================================

static bool emitObjectFile(
    Module& module,
    const fs::path& outputPath
) {

    InitializeNativeTarget();

    InitializeNativeTargetAsmPrinter();

    InitializeNativeTargetAsmParser();

    std::string triple =
        sys::getDefaultTargetTriple();

    module.setTargetTriple(
        triple
    );

    std::string error;

    const Target* target =
        TargetRegistry::lookupTarget(
            triple,
            error
        );

    if (!target) {

        errs()
            << "LLVM target lookup failed: "
            << error
            << "\n";

        return false;
    }

    TargetOptions options;

    auto machine =
        std::unique_ptr<TargetMachine>(
            target->createTargetMachine(
                triple,
                "generic",
                "",
                options,
                std::nullopt
            )
        );

    if (!machine) {

        errs()
            << "Could not create LLVM TargetMachine\n";

        return false;
    }

    module.setDataLayout(
        machine->createDataLayout()
    );

    std::error_code ec;

    raw_fd_ostream output(
        outputPath.string(),
        ec,
        sys::fs::OF_None
    );

    if (ec) {

        errs()
            << "Could not open object output: "
            << outputPath
            << "\n"
            << ec.message()
            << "\n";

        return false;
    }

    legacy::PassManager pass;

    if (machine->addPassesToEmitFile(
            pass,
            output,
            nullptr,
            CodeGenFileType::ObjectFile
        )) {

        errs()
            << "LLVM cannot emit object file\n";

        return false;
    }

    pass.run(module);

    output.flush();

    return true;
}

// ============================================================
// Usage
// ============================================================

static void printUsage() {

    std::cout
        << "Kyber Compiler\n"
        << "==============\n\n"

        << "Usage:\n"
        << "  kyberc <source.kyber>\n\n"

        << "Output:\n"
        << "  "
        << KYBER_OBJECT
        << "\n\n"

        << "Extern output:\n"
        << "  "
        << OBJECT_DIRECTORY
        << "\\kyber_extern_N.obj\n\n"

        << "n8n output:\n"
        << "  "
        << OBJECT_DIRECTORY
        << "\\kyber_n8n_N.obj\n\n"

        << "n8n blocks:\n"
        << "  #n8n \"workflow_name\"\n"
        << "  {\n"
        << "      url: \"https://your-n8n-host/webhook/...\"\n"
        << "      method: \"POST\"\n"
        << "      payload: \"{\\\"status\\\":\\\"ok\\\"}\"\n"
        << "  }\n\n"
        << "  Compiles to an int workflow_name(const char* payload)\n"
        << "  function that POSTs to the webhook via libcurl.\n"
        << "  Requires libcurl at compile and link time.\n\n"

        << "Ninja output:\n"
        << "  "
        << OBJECT_DIRECTORY
        << "\\kyber_ninja_N.obj\n\n"

        << "ninja blocks:\n"
        << "  #ninja \"workflow_name\"\n"
        << "  {\n"
        << "      <raw Ninja source>\n"
        << "  }\n\n"
        << "  Compiles to an int workflow_name(void) function that\n"
        << "  writes the embedded Ninja source to a temp file and\n"
        << "  runs it via \"python3 NINJA--COMPILER.py <file>\".\n"
        << "  Requires python3 and NINJA--COMPILER.py at RUNTIME,\n"
        << "  default path:\n"
        << "    "
        << NINJA_INTERPRETER_DEFAULT
        << "\n"
        << "  override with the KYBER_NINJA_INTERPRETER env var.\n";
}

// ============================================================
// Main
// ============================================================

int main(
    int argc,
    char** argv
) {

    if (argc < 2) {

        printUsage();

        return 1;
    }

    fs::path inputPath =
        argv[1];

    // --------------------------------------------------------
    // Read source.
    // --------------------------------------------------------

    std::string source =
        readFile(inputPath);

    if (source.empty()) {

        std::cerr
            << "Could not read Kyber source:\n"
            << inputPath
            << "\n";

        return 1;
    }

    // --------------------------------------------------------
    // Ensure .obj directory exists.
    // --------------------------------------------------------

    std::error_code ec;

    fs::create_directories(
        OBJECT_DIRECTORY,
        ec
    );

    if (ec) {

        std::cerr
            << "Could not create:\n"
            << OBJECT_DIRECTORY
            << "\n"
            << ec.message()
            << "\n";

        return 1;
    }

    // --------------------------------------------------------
    // Extract C/C++ extern blocks.
    // --------------------------------------------------------

    std::string afterExtern;

    std::vector<ExternBlock>
        externBlocks;

    if (!extractExternBlocks(
            source,
            afterExtern,
            externBlocks
        )) {

        std::cerr
            << "Failed to extract #extern blocks.\n";

        return 1;
    }

    // --------------------------------------------------------
    // Extract #n8n blocks.
    // --------------------------------------------------------

    std::string afterN8n;

    std::vector<N8nBlock>
        n8nBlocks;

    if (!extractN8nBlocks(
            afterExtern,
            afterN8n,
            n8nBlocks
        )) {

        std::cerr
            << "Failed to extract #n8n blocks.\n";

        return 1;
    }

    // --------------------------------------------------------
    // Extract #ninja blocks.
    // --------------------------------------------------------

    std::string kyberSource;

    std::vector<NinjaBlock>
        ninjaBlocks;

    if (!extractNinjaBlocks(
            afterN8n,
            kyberSource,
            ninjaBlocks
        )) {

        std::cerr
            << "Failed to extract #ninja blocks.\n";

        return 1;
    }

    // --------------------------------------------------------
    // Lex Kyber code.
    // --------------------------------------------------------

    KyberLexer lexer(
        kyberSource
    );

    auto tokens =
        lexer.tokenize();

    // --------------------------------------------------------
    // Parse Kyber code.
    // --------------------------------------------------------

    KyberParser parser(
        tokens
    );

    auto ast =
        parser.parse();

    if (!ast) {

        std::cerr
            << "Kyber parser failed.\n";

        return 1;
    }

    // --------------------------------------------------------
    // LLVM generation.
    // --------------------------------------------------------

    LLVMContext context;

    auto module =
        lowerToLLVM(
            ast.get(),
            context,
            "kyber_module"
        );

    // --------------------------------------------------------
    // Verify LLVM IR.
    // --------------------------------------------------------

    if (verifyModule(
            *module,
            &errs()
        )) {

        std::cerr
            << "LLVM verification failed.\n";

        return 1;
    }

    // --------------------------------------------------------
    // Generate Kyber object.
    // --------------------------------------------------------

    std::cout
        << "\n[Kyber LLVM]\n"
        << "Generating:\n"
        << KYBER_OBJECT
        << "\n";

    if (!emitObjectFile(
            *module,
            KYBER_OBJECT
        )) {

        std::cerr
            << "Could not generate Kyber object file.\n";

        return 1;
    }

    std::cout
        << "Generated: "
        << KYBER_OBJECT
        << "\n";

    // --------------------------------------------------------
    // Compile C/C++ extern blocks.
    // --------------------------------------------------------

    for (size_t i = 0;
         i < externBlocks.size();
         ++i) {

        fs::path externObject =
            OBJECT_DIRECTORY /
            (
                "kyber_extern_" +
                std::to_string(i) +
                ".obj"
            );

        if (!compileExternBlock(
                externBlocks[i],
                externObject,
                i
            )) {

            return 1;
        }
    }

    // --------------------------------------------------------
    // Compile #n8n blocks.
    // --------------------------------------------------------

    for (size_t i = 0;
         i < n8nBlocks.size();
         ++i) {

        fs::path n8nObject =
            OBJECT_DIRECTORY /
            (
                "kyber_n8n_" +
                std::to_string(i) +
                ".obj"
            );

        if (!compileN8nBlock(
                n8nBlocks[i],
                n8nObject,
                i
            )) {

            return 1;
        }
    }

    // --------------------------------------------------------
    // Compile #ninja blocks.
    // --------------------------------------------------------

    for (size_t i = 0;
         i < ninjaBlocks.size();
         ++i) {

        fs::path ninjaObject =
            OBJECT_DIRECTORY /
            (
                "kyber_ninja_" +
                std::to_string(i) +
                ".obj"
            );

        if (!compileNinjaBlock(
                ninjaBlocks[i],
                ninjaObject,
                i
            )) {

            return 1;
        }
    }

    // --------------------------------------------------------
    // IMPORTANT:
    //
    // The compiler DOES NOT LINK.
    //
    // kyberlink.exe is responsible for:
    //
    //   kyber.obj
    //   kyber_extern_0.obj, kyber_extern_1.obj, ...
    //   kyber_n8n_0.obj, kyber_n8n_1.obj, ...
    //   kyber_ninja_0.obj, kyber_ninja_1.obj, ...
    //   NForce libraries
    //   -lcurl (only if there are #n8n blocks)
    //
    // and producing the final .exe.
    //
    // #ninja objects don't need anything special at LINK time,
    // but at RUN time the machine executing the program needs
    // python3 and NINJA--COMPILER.py available (see
    // KYBER_NINJA_INTERPRETER / NINJA_INTERPRETER_DEFAULT).
    // --------------------------------------------------------

    std::cout
        << "\n====================================\n"
        << "Kyber compilation successful!\n"
        << "====================================\n"

        << "Object directory:\n"
        << OBJECT_DIRECTORY
        << "\n\n"

        << "Kyber object:\n"
        << KYBER_OBJECT
        << "\n"

        << "Extern objects generated: "
        << externBlocks.size()
        << "\n"

        << "n8n objects generated: "
        << n8nBlocks.size()
        << "\n"

        << "ninja objects generated: "
        << ninjaBlocks.size()
        << "\n";

    if (!n8nBlocks.empty()) {

        std::cout
            << "\nNote: #n8n blocks require libcurl.\n"
            << "Link the final executable with -lcurl.\n";
    }

    if (!ninjaBlocks.empty()) {

        std::cout
            << "\nNote: #ninja blocks require python3 and\n"
            << "NINJA--COMPILER.py at RUNTIME (not link time).\n"
            << "Default interpreter path:\n"
            << "  "
            << NINJA_INTERPRETER_DEFAULT
            << "\n"
            << "Override with KYBER_NINJA_INTERPRETER if it's\n"
            << "installed elsewhere.\n";
    }

    std::cout
        << "\nNext stage:\n"
        << "kyberlink.exe\n";

    return 0;
}
