use std::io;
use anyhow::Result;

use ureq;
use std::time::Duration;

use serde::{Serialize, Deserialize};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
//use ratatui_macros::span;
use ratatui_macros::line;
use ratatui::{
    buffer::Buffer,
    layout::{Rect, Constraint, Alignment},
    style::{Color, Style, Stylize},
    symbols::border,
    text::{Span, Line, Text},
    widgets::{Block, Borders, Paragraph, Wrap, Widget},
    DefaultTerminal, Frame,
};

#[derive(Debug, Default)]
struct App {
    history: Vec<String>,
    prompt: String,
    exit: bool,
}

fn main() -> io::Result<()> {
    let mut app = App::default();
    app.prompt = "".to_string();
    app.history.push("Hello, what do you want?".to_string());
    app.history.push("Hello Laura Maxwell".to_string());
    app.history.push("Hello Quickfix".to_string());
    ratatui::run(|terminal| app.run(terminal));

    Ok(())
}

#[derive(Serialize)]
struct MySendBody {
   prompt: String,
}

#[derive(Deserialize)]
struct MyRecvBody {
   answer: String,
}

fn ask(question: &str) -> Result<String> {
    let send_body = MySendBody { prompt: question.to_string() };
    let recv_body = ureq::post("http://0.0.0.0:3000/")
	.header("Accept", "application/json")
	.send_json(&send_body)?
	.body_mut()
	.read_json::<MyRecvBody>()?;

    Ok(recv_body.answer)
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area());
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Esc => self.exit(),
            KeyCode::Backspace => {
                self.prompt.pop();
            },
            KeyCode::Enter => {
                self.history.push(self.prompt.to_string());
                self.history.push("Loading question".to_string());
                self.prompt.clear();

                let answer = ask(&self.prompt).unwrap_or("".to_string());
                self.history.push(self.prompt.to_string());
                self.history.push(answer.to_string());
            },
            KeyCode::Char(c) => {
                self.prompt.push(c);
            },
            _ => {},
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Line::from(" PubNub Search ".bold());
        let instructions = Line::from(vec![
            " <Enter>".blue().bold(),
            " Submit".into(),
            " <Esc>".blue().bold(),
            " Quit ".into(),
        ]);
        let block = Block::bordered()
            .title(title.centered())
            .border_set(border::THICK);

        let height: i32 = area.height as i32;
        let length: i32 = self.history.len() as i32;
        let offset: i32 = 5;
        let start: usize = *vec![
            0, length - height + offset,
        ].iter().max().unwrap() as usize;

        let messages: Vec<Line> = self.history[start..length as usize]
            .iter()
            .map(|string| Line::from(string.to_string()))
            .collect();

        let chat_history = Text::from(messages).on_magenta().dim().bold();
        Paragraph::new(chat_history)
            .block(block)
            .wrap(Wrap { trim: true })
            .render(area, buf);

        let text = self.prompt.to_string();
        let prompt = Text::from(vec![line![
            text, " ".on_cyan()
        ]]);
        let prompt_area = Rect {
            x: area.x,
            y: area.height - 4,
            width: area.width,
            height: area.height,
        };

        let prompt_block = Block::bordered()
            .title_bottom(instructions.centered())
            .border_set(border::THICK);

        Paragraph::new(prompt)
            .block(prompt_block)
            .wrap(Wrap { trim: true })
            .render(prompt_area, buf);
    }
}